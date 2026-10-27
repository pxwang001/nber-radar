import dataclasses
import os
import re

import streamlit as st

from nber_radar import (
    CACHE_MAX_AGE,
    PROGRAM_CODES,
    NberDataError,
    QueryUnderstanding,
    QueryGroup,
    clear_cache,
    decompose_chinese_query,
    understand_query,
    data_freshness_notice,
    fetch_nber_papers,
    apply_translations,
    translate_paper,
    save_translation,
    LlmError,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    list_feedback,
    load_api_config,
    load_papers_cache,
    papers_to_dataframe,
    parse_edited_groups,
    recommend_public_papers,
    render_markdown_digest,
    papers_to_ris,
    papers_to_bibtex,
    save_api_config,
    save_feedback,
    search_nber_papers,
    list_programs,
    _get_search_index,
    save_topic,
    list_topics,
    load_topic,
    delete_topic,
    mark_pushed,
    list_pushed_ids,
)

st.set_page_config(page_title="NBER 公共经济与管理雷达", layout="wide")

saved_cfg = load_api_config()
for key, default in (
    ("papers", []),
    ("search_mode", ""),
    ("custom_kws", []),
    ("data_notice", ""),
    ("auto_translate_pending", False),
    ("translate_error", ""),
    ("query_glossary", {}),
    ("api_key", os.getenv("NBER_RADAR_API_KEY", "")),
    ("api_base_url", saved_cfg.get("api_base_url") or DEFAULT_LLM_BASE_URL),
    ("api_model", saved_cfg.get("api_model") or DEFAULT_LLM_MODEL),
    ("current_groups", []),
    ("current_year_from", None),
    ("current_year_to", None),
    ("current_programs", None),
):
    if key not in st.session_state:
        st.session_state[key] = default

st.title("NBER 公共经济与管理雷达")
st.caption("自动推荐相关 Working Paper，也支持中文关键词检索。")

with st.sidebar:
    st.header("查找论文")
    st.caption("默认优先推荐公共经济、公共管理及相关政策研究。")

    result_count = st.slider("显示数量", 10, 100, 30, step=10)
    min_score = st.slider(
        "最低相关度",
        10,
        60,
        25,
        step=5,
        help="数值越高，结果越少，但通常更准确。",
    )

    prog_list = list_programs()
    prog_options = {f"{code} - {name}": code for code, name in prog_list}
    selected_progs_display = st.multiselect(
        "NBER 分类筛选",
        options=list(prog_options.keys()),
        default=[],
        help="不选表示不限，多选为 OR 关系。",
    )
    selected_programs = [prog_options[p] for p in selected_progs_display] if selected_progs_display else None

    try:
        index = _get_search_index()
        if not index.empty and "issue_date" in index.columns:
            years = index["issue_date"].dt.year
            min_year = int(years.min())
            max_year = int(years.max())
            year_range = st.slider(
                "年份范围",
                min_year, max_year, (min_year, max_year),
                help="依据发表日期筛选。",
            )
            year_from, year_to = year_range
        else:
            year_from, year_to = None, None
    except Exception:
        year_from, year_to = None, None

    with st.form("search_form"):
        custom_query = st.text_input(
            "关键词",
            placeholder="例如：地方政府债务、人工智能监管、教育政策",
        )
        custom_search = st.form_submit_button("搜索全库", type="primary")

    recommend_latest = st.button("推荐最新论文", use_container_width=True)

    cached = load_papers_cache()
    if cached:
        st.caption(f"已有缓存：{len(cached)} 篇，{CACHE_MAX_AGE // 3600}小时内直接复用。")

    with st.expander("刷新数据"):
        st.caption("清除旧缓存，并重新获取 NBER 数据。")
        force_refresh = st.button("立即刷新", use_container_width=True)

    with st.expander("我的主题"):
        st.caption("保存常用的检索参数组合，一键运行。")
        try:
            topics = list_topics()
            if topics:
                for topic_id, topic_name, created_at in topics:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(topic_name)
                    with col2:
                        if st.button("运行", key=f"run_{topic_id}", use_container_width=True):
                            topic_data = load_topic(topic_id)
                            if topic_data:
                                name, understanding, year_from_saved, year_to_saved, programs_saved = topic_data
                                with st.spinner(f"正在搜索：{name}……"):
                                    try:
                                        results = search_nber_papers(
                                            understanding,
                                            max_results=result_count,
                                            min_score=min_score,
                                            year_from=year_from_saved,
                                            year_to=year_to_saved,
                                            programs=programs_saved if programs_saved else None,
                                        )
                                        set_results(results, f"主题：{name}", [], notice="")
                                        st.success(f"已运行主题：{name}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"搜索失败：{e}")
                    with col3:
                        if st.button("删除", key=f"del_{topic_id}", use_container_width=True):
                            delete_topic(topic_id)
                            st.success("已删除")
                            st.rerun()
            else:
                st.caption("暂无已保存的主题。")
        except Exception as e:
            st.warning(f"读取主题失败：{e}")

    with st.expander("AI 中文摘要设置"):
        st.caption("API Key 只保留在当前页面，刷新后需重新填写。")
        st.text_input("API Key", type="password", key="api_key")
        st.text_input("API 地址", key="api_base_url")
        st.text_input("模型名", key="api_model")
        st.caption("推荐 glm-4-flash-250414（免费、较快）。高峰期 glm-4.7-flash 常被限流。")
        auto_n = st.slider("搜索后自动翻译前几篇", 0, 20, 5, help="免费模型每篇约 5–9 秒，翻译结果会永久缓存在本机。")
        if st.button("保存模型地址和名称"):
            save_api_config(
                api_base_url=st.session_state.api_base_url,
                api_model=st.session_state.api_model,
            )
            st.success("已保存模型地址和名称。API Key 只保留在当前页面。")


def set_results(papers, mode, keywords=None, notice=""):
    st.session_state.papers = apply_translations(papers)
    st.session_state.auto_translate_pending = True
    st.session_state.search_mode = mode
    st.session_state.custom_kws = keywords or []
    st.session_state.data_notice = notice


if force_refresh:
    clear_cache(include_tsv=True)
    try:
        with st.spinner("正在重新获取 NBER 数据……"):
            raw = fetch_nber_papers(max_papers=400, force_refresh=True)
            results = recommend_public_papers(
                raw,
                max_results=result_count,
                min_score=min_score,
            )
        set_results(results, "最新公共经济与管理推荐", notice=data_freshness_notice(raw))
        if not results:
            st.warning("数据已刷新，但暂时没有达到当前相关度要求的论文。")
    except Exception as exc:
        st.error(f"刷新失败：{exc}")

elif recommend_latest:
    try:
        with st.spinner("正在整理最新论文……"):
            raw = fetch_nber_papers(max_papers=400)
            results = recommend_public_papers(
                raw,
                max_results=result_count,
                min_score=min_score,
            )
        set_results(results, "最新公共经济与管理推荐", notice=data_freshness_notice(raw))
    except NberDataError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"读取论文失败：{exc}")

elif custom_search:
    if not custom_query.strip():
        st.warning("请先输入关键词。")
    else:
        try:
            has_llm = bool(st.session_state.api_key.strip())
            with st.spinner("正在全库搜索并排序……"):
                understanding = understand_query(
                    custom_query,
                    api_key=st.session_state.api_key if has_llm else "",
                    base_url=st.session_state.api_base_url,
                    model=st.session_state.api_model,
                )
                results = search_nber_papers(
                    understanding,
                    max_results=result_count,
                    min_score=min_score,
                    year_from=year_from,
                    year_to=year_to,
                    programs=selected_programs,
                )
            en_kws = []
            for group in understanding.groups:
                en_kws.extend(group.terms)
            st.session_state.query_glossary = {
                g.terms[0].lower(): g.concept for g in understanding.groups
                if g.terms and g.concept and g.concept != "query" and re.search(r"[一-鿿]", g.concept)
            }
            st.session_state.current_groups = understanding.groups
            st.session_state.current_year_from = year_from
            st.session_state.current_year_to = year_to
            st.session_state.current_programs = selected_programs
            set_results(results, f"关键词搜索：{custom_query}", en_kws, notice="")

            if understanding.error:
                st.warning(understanding.error)
            if understanding.uncovered:
                uncovered_str = "、".join(understanding.uncovered)
                msg = f"以下词未能识别：{uncovered_str}。词典模式下词量有限，在左侧填写 API Key 可大幅提高识别率"
                st.warning(msg)
            if understanding.groups:
                concept_text = " × ".join(g.concept for g in understanding.groups if g.concept != "query")
                if concept_text:
                    st.caption(f"已理解为：{concept_text}（{'大模型拆词' if understanding.source == 'llm' else '内置词典拆词'}）")
                for group in understanding.groups:
                    if group.concept != "query":
                        terms_text = "；".join(group.terms[:4])
                        st.caption(f"{group.concept}：{terms_text}")

                with st.expander("调整检索词"):
                    st.caption("编辑每个概念组的英文词（逗号分隔），或删除不需要的组。")
                    edit_group_texts = []
                    for i, group in enumerate(understanding.groups):
                        if group.concept == "query":
                            continue
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            edited = st.text_input(
                                f"概念组 {len(edit_group_texts) + 1}: {group.concept}",
                                value=", ".join(group.terms),
                                key=f"edit_group_{i}",
                                label_visibility="collapsed",
                            )
                            edit_group_texts.append(edited)
                        with col2:
                            delete = st.checkbox("删除", key=f"del_group_{i}")
                            if delete:
                                edit_group_texts[-1] = ""

                    new_concept = st.text_input(
                        "新增概念组（英文词，逗号分隔）",
                        key="new_concept_input",
                        label_visibility="collapsed",
                    )

                    if st.button("按修改后的词重搜", key="research_btn", type="primary"):
                        try:
                            edited_groups = parse_edited_groups(edit_group_texts, new_concept)
                            if not edited_groups:
                                st.warning("修改后没有有效词，请检查输入。")
                            else:
                                with st.spinner("按修改后的词重新搜索……"):
                                    edited_understanding = QueryUnderstanding(
                                        groups=edited_groups,
                                        uncovered=[],
                                        source="手动调整",
                                    )
                                    new_results = search_nber_papers(
                                        edited_understanding,
                                        max_results=result_count,
                                        min_score=min_score,
                                        year_from=year_from,
                                        year_to=year_to,
                                        programs=selected_programs,
                                    )
                                set_results(new_results, f"手动调整搜索：{custom_query}",
                                           keywords=[t for g in edited_groups for t in g.terms], notice="")
                                st.rerun()
                        except Exception as exc:
                            st.error(f"调整搜索失败：{exc}")

            if not understanding.groups:
                st.warning("没有识别出有效关键词，可以换一个更具体的说法。")
            elif not results:
                terms_shown = "；".join(t for g in understanding.groups for t in g.terms)
                relaxed = search_nber_papers(
                    understanding, max_results=result_count, min_score=0, use_feedback=False,
                    year_from=year_from, year_to=year_to, programs=selected_programs,
                )
                if relaxed:
                    st.warning(f"当前相关度要求下没有结果，把“最低相关度”调到 10 以下可得到 {len(relaxed)} 篇。")
                else:
                    st.warning(f"NBER 库中没有命中这些英文词：{terms_shown}。可能该主题本身很少被 NBER 论文覆盖，也可以在“调整检索词”里换英文词再试。")
        except NberDataError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"搜索失败：{exc}")

def run_translation(target_ids, label):
    """逐篇翻译并写入缓存，遇到模型错误立即停止并记录原因。"""
    index = {p.nber_id: i for i, p in enumerate(st.session_state.papers)}
    bar = st.progress(0, text=label)
    st.session_state.translate_error = ""
    for k, pid in enumerate(target_ids):
        paper = st.session_state.papers[index[pid]]
        try:
            tr = translate_paper(
                paper, st.session_state.api_key,
                st.session_state.api_base_url, st.session_state.api_model,
                glossary=st.session_state.query_glossary,
            )
            save_translation(tr)
        except LlmError as exc:
            st.session_state.translate_error = f"第 {k + 1} 篇翻译失败：{exc}。已完成的内容已保存，稍后可继续。"
            break
        st.session_state.papers[index[pid]] = dataclasses.replace(
            paper, cn_title=tr.title_zh, cn_abstract=tr.abstract_zh, cn_summary=tr.one_liner,
        )
        bar.progress((k + 1) / len(target_ids), text=f"{label} {k + 1}/{len(target_ids)}")
    bar.empty()


if st.session_state.auto_translate_pending:
    st.session_state.auto_translate_pending = False
    if st.session_state.api_key and auto_n > 0:
        todo = [p.nber_id for p in st.session_state.papers[:auto_n] if not p.cn_title and p.title]
        if todo:
            run_translation(todo, "正在翻译前几篇")

papers = st.session_state.papers

if st.session_state.translate_error:
    st.warning(st.session_state.translate_error)

if st.session_state.search_mode:
    st.info(f"当前结果：{st.session_state.search_mode}")
    if st.session_state.data_notice:
        st.warning(st.session_state.data_notice)
    if st.session_state.custom_kws:
        st.caption("实际检索词：" + "；".join(st.session_state.custom_kws))
    filter_parts = []
    try:
        if year_from is not None and year_to is not None:
            idx = _get_search_index()
            if not idx.empty and "issue_date" in idx.columns:
                min_year = int(idx["issue_date"].dt.year.min())
                max_year = int(idx["issue_date"].dt.year.max())
                if year_from != min_year or year_to != max_year:
                    filter_parts.append(f"年份：{year_from}–{year_to}")
    except Exception:
        pass
    if selected_programs:
        prog_names = [f"{code}({PROGRAM_CODES.get(code, code)})" for code in selected_programs]
        filter_parts.append(f"分类：{', '.join(prog_names)}")
    if filter_parts:
        st.caption("生效筛选：" + "；".join(filter_parts))

if papers:
    df = papers_to_dataframe(papers)
    c1, c2, c3 = st.columns(3)
    c1.metric("论文数量", len(df))
    c2.metric("高相关", int(df["level"].isin(["A", "B"]).sum()))
    c3.metric("已有中文译文", sum(bool(p.cn_title) for p in papers))

    if papers and papers[0].groups_total > 0:
        fully_matched = sum(1 for p in papers if p.groups_matched == p.groups_total)
        if fully_matched < len(papers):
            msg = f"同时涉及全部概念的有 {fully_matched} 篇，其余为只命中部分概念"
            st.info(msg)

    left, right = st.columns([3, 2])
    with left:
        md = render_markdown_digest(papers, title="NBER 公共经济与管理论文")
        ris = papers_to_ris(papers)
        bibtex = papers_to_bibtex(papers)
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button(
            "下载 Markdown",
            data=md,
            file_name="nber_public_management_digest.md",
            mime="text/markdown",
            use_container_width=True,
        )
        d2.download_button(
            "下载 CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="nber_public_management_papers.csv",
            mime="text/csv",
            use_container_width=True,
        )
        d3.download_button(
            "下载 RIS",
            data=ris.encode("utf-8"),
            file_name="nber_public_management_papers.ris",
            mime="text/plain",
            use_container_width=True,
        )
        d4.download_button(
            "下载 BibTeX",
            data=bibtex.encode("utf-8"),
            file_name="nber_public_management_papers.bib",
            mime="text/plain",
            use_container_width=True,
        )

    with right:
        untranslated = [p.nber_id for p in papers if not p.cn_title and p.title]
        if st.session_state.api_key and untranslated:
            if st.button(f"翻译其余 {len(untranslated)} 篇（约 {max(1, len(untranslated) * 7 // 60)} 分钟）", type="primary"):
                run_translation(untranslated, "正在翻译")
                st.rerun()
        elif not st.session_state.api_key:
            st.caption("填写左侧 API Key 后，可自动翻译题目、摘要并生成一句话。")

    st.subheader("论文结果")
    for idx, paper in enumerate(papers):
        with st.container(border=True):
            main, actions = st.columns([5, 1])
            with main:
                st.markdown(f"### {idx + 1}. [{paper.cn_title or paper.title}]({paper.url})")
                if paper.cn_title:
                    st.caption(paper.title)
                caption_parts = [f"{paper.level}级", f"相关度 {paper.score}/100"]
                if paper.groups_total > 0:
                    caption_parts.append(f"命中 {paper.groups_matched}/{paper.groups_total} 个概念")
                caption_parts.extend([
                    paper.published or "日期未知",
                    paper.nber_id or "编号未知"
                ])
                st.caption(" · ".join(caption_parts))
                st.caption(f"作者：{paper.authors or '未知'}")
                if paper.programs:
                    st.caption(f"NBER 分类：{paper.programs}")
                if paper.cn_summary:
                    st.markdown(f"**一句话：** {paper.cn_summary}")
                st.caption(f"命中原因：{paper.reason}")

                labels = list(paper.cn_highlights or paper.highlights)
                if labels:
                    st.markdown("　".join(f"`{item}`" for item in labels[:3]))
                if paper.abstract:
                    with st.expander("查看摘要"):
                        if paper.cn_abstract:
                            zh_tab, en_tab = st.tabs(["中文", "English"])
                            zh_tab.write(paper.cn_abstract)
                            en_tab.write(paper.abstract)
                        else:
                            st.write(paper.abstract)

            with actions:
                if st.session_state.api_key and not paper.cn_title and paper.title:
                    if st.button("翻译", key=f"tr_{paper.nber_id}_{idx}"):
                        run_translation([paper.nber_id], "正在翻译")
                        st.rerun()

                if st.button("收藏", key=f"save_{paper.nber_id}_{idx}"):
                    save_feedback(paper, "saved")
                    st.success("已收藏")

                if st.button("忽略", key=f"ignore_{paper.nber_id}_{idx}"):
                    save_feedback(paper, "ignored")
                    st.session_state.papers = [p for p in papers if p.nber_id != paper.nber_id]
                    st.rerun()

    save_col, run_col = st.columns([3, 1])
    with save_col:
        topic_name = st.text_input("主题名称", placeholder="例如：人工智能与就业政策")
    with run_col:
        if st.button("保存为主题", use_container_width=True):
            if topic_name.strip() and st.session_state.current_groups:
                try:
                    save_topic(
                        topic_name.strip(),
                        st.session_state.current_groups,
                        st.session_state.current_year_from,
                        st.session_state.current_year_to,
                        st.session_state.current_programs,
                    )
                    st.success(f"已保存主题：{topic_name.strip()}")
                except Exception as e:
                    st.error(f"保存失败：{e}")
            elif not st.session_state.current_groups:
                st.warning("请先进行搜索。")
            else:
                st.warning("请输入主题名称。")

    with st.expander("查看表格"):
        st.dataframe(df, use_container_width=True)
else:
    st.info("输入关键词搜索，或点击左侧推荐最新论文。")

with st.expander("我的收藏和忽略记录"):
    try:
        feedback = list_feedback()
        if feedback.empty:
            st.caption("暂无记录。")
        else:
            display = feedback.copy()
            display["action"] = display["action"].map({"saved": "收藏", "ignored": "忽略"}).fillna(display["action"])
            st.dataframe(
                display[["title", "action", "created_at"]],
                use_container_width=True,
                hide_index=True,
            )
    except Exception as exc:
        st.warning(f"反馈记录暂时无法读取：{exc}")
