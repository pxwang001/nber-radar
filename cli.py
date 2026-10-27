import argparse

from nber_radar import (
    fetch_nber_papers,
    data_freshness_notice,
    recommend_public_papers,
    render_markdown_digest,
    save_outputs,
    send_email_digest,
)


def main():
    parser = argparse.ArgumentParser(description="NBER 公共经济与管理论文周报")
    parser.add_argument("--max-papers", type=int, default=50, help="最终输出数量")
    parser.add_argument("--min-score", type=int, default=25, help="最低相关度")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--send-email", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="强制刷新数据")
    args = parser.parse_args()

    raw = fetch_nber_papers(max_papers=max(300, args.max_papers * 5), force_refresh=args.refresh)
    notice = data_freshness_notice(raw)
    if notice:
        print(f"提示：{notice}")

    papers = recommend_public_papers(
        raw,
        max_results=args.max_papers,
        min_score=args.min_score,
    )
    csv_path, md_path = save_outputs(papers, out_dir=args.out_dir)
    print(f"筛选出 {len(papers)} 篇")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")

    if args.send_email:
        body = render_markdown_digest(papers, title="NBER 公共经济与管理论文")
        send_email_digest(body)
        print("邮件已发送")


if __name__ == "__main__":
    main()
