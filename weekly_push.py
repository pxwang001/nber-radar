#!/usr/bin/env python3
import argparse
import subprocess
import sys
import datetime as dt
from pathlib import Path
from typing import Callable, List, Optional
import sqlite3

from nber_radar import (
    fetch_nber_papers,
    clear_cache,
    list_topics,
    load_topic,
    search_nber_papers,
    list_pushed_ids,
    mark_pushed,
    DEFAULT_DB_PATH,
)


def send_via_cc_connect(message: str) -> int:
    """Send message via cc-connect. Returns subprocess return code."""
    try:
        proc = subprocess.run(
            ["cc-connect", "send", "--stdin"],
            input=message.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        return proc.returncode
    except FileNotFoundError:
        print("错误: cc-connect 命令未找到，请确保已安装。", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("错误: cc-connect 发送超时。", file=sys.stderr)
        return 1


def collect_weekly_papers(
    days: int = 7,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict:
    """Collect papers from all saved topics within recent N days.

    Returns:
        dict: {topic_id: (topic_name, papers_list)}
    """
    result = {}
    topics = list_topics(db_path)
    cutoff = dt.date.today() - dt.timedelta(days=days)

    for topic_id, name, _ in topics:
        loaded = load_topic(topic_id, db_path)
        if not loaded:
            continue

        name, understanding, year_from, year_to, programs = loaded
        pushed_ids = list_pushed_ids(topic_id, db_path)

        papers = search_nber_papers(
            understanding,
            max_results=500,
            use_feedback=False,
            year_from=year_from,
            year_to=year_to,
            programs=programs or None,
        )

        new_papers = [
            p for p in papers
            if p.nber_id not in pushed_ids
            and p.published
            and dt.datetime.fromisoformat(p.published).date() >= cutoff
        ]

        if new_papers:
            result[topic_id] = (name, new_papers)

    return result


def format_message_for_topic(name: str, papers: List) -> List[str]:
    """Format papers for a topic into message chunks (~3000 chars each).

    Returns:
        list of message strings
    """
    if not papers:
        return []

    lines = [f"【{name}】本周新增 {len(papers)} 篇"]

    for i, p in enumerate(papers, 1):
        title = p.cn_title or p.title
        nber_id = p.nber_id.upper()
        date_str = p.published[:10] if p.published else "日期未知"

        lines.append(f"{i}. {title}")
        lines.append(f"   ({nber_id}, {date_str})")
        lines.append(f"   {p.url}")

    def split_long_line(line: str, max_len: int = 3000) -> List[str]:
        """Recursively split line if it exceeds max_len."""
        if len(line) <= max_len:
            return [line]
        mid = max_len // 2
        return split_long_line(line[:mid], max_len) + split_long_line(line[mid:], max_len)

    messages = []
    current = ""
    for line in lines:
        line_parts = split_long_line(line)
        for part in line_parts:
            if len(current) + len(part) + 1 > 3000:
                if current:
                    messages.append(current)
                current = part
            else:
                current = current + "\n" + part if current else part

    if current:
        messages.append(current)

    return messages


def push_weekly(
    dry_run: bool = False,
    days: int = 7,
    no_refresh: bool = False,
    topic_name: Optional[str] = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    send_func: Optional[Callable[[str], int]] = None,
) -> int:
    """Main weekly push logic.

    Args:
        dry_run: Only print, don't send or mark_pushed.
        days: Look for papers in last N days.
        no_refresh: Skip refreshing NBER data.
        topic_name: Only push for this topic (optional).
        db_path: Path to SQLite database.
        send_func: Function to send message (for testing).

    Returns:
        0 on success, non-zero on error.
    """
    if send_func is None:
        send_func = send_via_cc_connect

    try:
        if not no_refresh:
            print("正在刷新 NBER 数据...", file=sys.stderr)
            fetch_nber_papers(force_refresh=True)
    except Exception as e:
        print(f"错误: 刷新 NBER 数据失败: {e}", file=sys.stderr)
        return 1

    try:
        collected = collect_weekly_papers(days, db_path)
    except Exception as e:
        print(f"错误: 收集论文失败: {e}", file=sys.stderr)
        return 1

    if topic_name:
        filtered = {tid: (name, papers) for tid, (name, papers) in collected.items() if name == topic_name}
        if not filtered:
            print(f"主题 '{topic_name}' 无新论文或不存在。", file=sys.stderr)
            return 0
        collected = filtered

    if not collected:
        print("本周无新增论文。", file=sys.stderr)
        return 0

    all_messages = []
    topic_papers = {}

    for topic_id, (name, papers) in collected.items():
        print(f"【{name}】新增 {len(papers)} 篇", file=sys.stderr)
        topic_papers[topic_id] = papers
        messages = format_message_for_topic(name, papers)
        all_messages.extend(messages)

    if not all_messages:
        print("没有生成消息。", file=sys.stderr)
        return 0

    for msg in all_messages:
        print("\n--- 消息预览 ---", file=sys.stderr)
        print(msg, file=sys.stderr)

        if dry_run:
            print("[DRY-RUN] 未发送此消息", file=sys.stderr)
        else:
            ret = send_func(msg)
            if ret != 0:
                print(f"错误: 发送消息失败 (返回码 {ret})", file=sys.stderr)
                return ret
            print("✓ 消息已发送", file=sys.stderr)

    if not dry_run:
        for topic_id, papers in topic_papers.items():
            nber_ids = [p.nber_id for p in papers]
            try:
                mark_pushed(topic_id, nber_ids, db_path)
                print(f"✓ 标记 {len(nber_ids)} 篇为已推送", file=sys.stderr)
            except Exception as e:
                print(f"错误: 标记推送失败: {e}", file=sys.stderr)
                return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="周推送：发送最近一周新发布的论文到微信。"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印消息，不发送也不标记为已推送。",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="查找最近 N 天内的论文（默认 7 天）。",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="跳过刷新 NBER 数据。",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="只推送指定的主题名称（可选）。",
    )

    args = parser.parse_args()

    ret = push_weekly(
        dry_run=args.dry_run,
        days=args.days,
        no_refresh=args.no_refresh,
        topic_name=args.topic,
    )

    sys.exit(ret)


if __name__ == "__main__":
    main()
