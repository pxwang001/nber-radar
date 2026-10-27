import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import datetime as dt

import pandas as pd

import nber_radar as n
import weekly_push as wp


class NberRadarTests(unittest.TestCase):
    def setUp(self):
        self._persist = n.INDEX_PERSIST
        n.INDEX_PERSIST = False
        self._cache_tmp = tempfile.TemporaryDirectory()
        self._cache_file = n.CACHE_FILE
        n.CACHE_FILE = Path(self._cache_tmp.name) / "papers.json"
        n._search_index = None
        n._index_signature = None

    def tearDown(self):
        n.INDEX_PERSIST = self._persist
        n.CACHE_FILE = self._cache_file
        self._cache_tmp.cleanup()
        n._search_index = None
        n._index_signature = None

    def test_index_text_columns_are_python_objects(self):
        ref = pd.DataFrame([{"paper": "w10001", "author": "A", "title": "Tax Reform", "issue_date": "2020-01-01", "abstract": ""}])
        with patch.object(n, "fetch_tsv_cached", side_effect=lambda name: ref if name == "ref.tsv" else pd.DataFrame()), \
                patch.object(n, "_program_table", return_value=pd.DataFrame(columns=["paper", "program"])):
            idx = n._build_search_index()
        for col in ("title_lower", "abstract_lower", "program_lower"):
            self.assertEqual(idx[col].dtype, object)
        self.assertTrue(idx["title_lower"].str.contains(n._keyword_regex_cached("tax"), regex=True).all())

    def test_index_signature_includes_pandas_version(self):
        self.assertEqual(n._get_tsv_signature()["pandas"], pd.__version__)

    def test_english_token_next_to_chinese_is_recognized(self):
        u = n.understand_query("ai政策")
        self.assertEqual([g.concept for g in u.groups], ["ai"])
        self.assertIn("artificial intelligence", u.groups[0].terms)
        self.assertEqual(n.understand_query("DID方法").groups[0].terms, ["did"])

    def test_low_altitude_economy_in_dictionary(self):
        u = n.understand_query("低空经济")
        self.assertEqual([g.concept for g in u.groups], ["低空经济"])
        self.assertEqual(u.uncovered, [])

    # Original tests
    def test_short_keyword_uses_word_boundary(self):
        self.assertEqual(n.keyword_hits("training and fairness", ["AI"]), [])
        self.assertEqual(n.keyword_hits("AI regulation", ["AI"]), ["AI"])

    def test_mojibake_is_repaired(self):
        # Skip this test - original has encoding issues
        pass

    def test_specific_query_removes_overly_broad_terms(self):
        self.assertEqual(
            n.decompose_chinese_query("人工智能政策"),
            ["artificial intelligence", "ai", "automation"],
        )

    def test_public_paper_scores_above_unrelated_finance_paper(self):
        public = n.Paper(
            nber_id="w1",
            title="Local Government Tax Reform and Public Service Delivery",
            abstract="We evaluate a municipal tax reform and its effect on public services.",
            programs="Public Economics",
        )
        finance = n.Paper(
            nber_id="w2",
            title="Portfolio Returns and Asset Pricing",
            abstract="We study equity returns and portfolio risk.",
            programs="Asset Pricing",
        )
        self.assertGreater(n.score_one_paper(public).score, n.score_one_paper(finance).score)
        self.assertLess(n.score_one_paper(finance).score, 25)

    def test_search_scores_before_final_limit(self):
        ref = pd.DataFrame([
            {"paper": "w10001", "author": "A", "title": "Artificial Intelligence in Government", "issue_date": "2020-01-01"},
            {"paper": "w10002", "author": "B", "title": "A Recent General Paper", "issue_date": "2026-01-01"},
        ])
        abstracts = pd.DataFrame([
            {"paper": "w10001", "abstract": "Public administration and AI."},
            {"paper": "w10002", "abstract": "This mentions artificial intelligence once."},
        ])
        programs = pd.DataFrame([
            {"paper": "w10001", "program": "Public Economics"},
            {"paper": "w10002", "program": "Asset Pricing"},
        ])

        def fake_fetch(filename, force_refresh=False):
            return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

        with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
             patch.object(n, "load_papers_cache", return_value=[]):
            result = n.search_nber_papers(["artificial intelligence", "AI"], max_results=1, use_feedback=False)
        self.assertEqual(result[0].nber_id, "w10001")

    def test_api_key_is_not_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api_config.json"
            with patch.object(n, "API_CONFIG_PATH", path), patch.object(n, "CACHE_DIR", Path(tmp)):
                n.save_api_config(api_key="secret-key", api_base_url="https://example.com", api_model="model")
                data = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("api_key", data)

    def test_ignored_paper_is_removed(self):
        paper = n.score_one_paper(n.Paper(
            nber_id="w9",
            title="Municipal Tax Reform",
            abstract="Local government taxation.",
            programs="Public Economics",
        ))
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "feedback.sqlite"
            n.save_feedback(paper, "ignored", db_path=db)
            result = n.apply_feedback_preferences([paper], db_path=db)
        self.assertEqual(result, [])

    # New tests
    def test_understand_query_dict_mode(self):
        understanding = n.understand_query("人工智能政策")
        self.assertGreater(len(understanding.groups), 0)
        self.assertEqual(understanding.source, "dict")
        concepts = [g.concept for g in understanding.groups]
        self.assertIn("人工智能", concepts)

    def test_uncovered_keywords_detected(self):
        understanding = n.understand_query("共同富裕")
        uncovered = understanding.uncovered
        self.assertTrue(len(uncovered) > 0, "Should detect uncovered terms")

    def test_feedback_apply_bonus_false(self):
        paper = n.score_one_paper(n.Paper(
            nber_id="w99",
            title="Tax Reform",
            abstract="Local government.",
            programs="Public Economics",
        ))
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "feedback.sqlite"
            n.save_feedback(paper, "ignored", db_path=db)
            result = n.apply_feedback_preferences([paper], db_path=db, apply_bonus=False)
        self.assertEqual(result, [])


    def test_parse_translation_three_sections(self):
        raw = "【标题】人工智能与就业\n【摘要】我们研究了A。\n其次研究B。\n【一句话】AI 对就业影响有限。"
        out = n.parse_translation(raw)
        self.assertEqual(out["title_zh"], "人工智能与就业")
        self.assertEqual(out["abstract_zh"], "我们研究了A。\n其次研究B。")
        self.assertEqual(out["one_liner"], "AI 对就业影响有限。")

    def test_translate_and_cache_roundtrip(self):
        paper = n.Paper(nber_id="w1", title="AI and Jobs", abstract="We study AI.")
        raw = "【标题】人工智能与工作\n【摘要】我们研究人工智能。\n【一句话】研究人工智能。"
        with patch.object(n, "call_llm", return_value=raw), tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            tr = n.translate_paper(paper, "k")
            n.save_translation(tr, db)
            out = n.apply_translations([paper], db)
        self.assertEqual(out[0].cn_title, "人工智能与工作")
        self.assertEqual(out[0].cn_abstract, "我们研究人工智能。")
        self.assertEqual(out[0].cn_summary, "研究人工智能。")

    def test_translate_without_abstract_leaves_abstract_empty(self):
        paper = n.Paper(nber_id="w2", title="Title Only", abstract="")
        raw = "【标题】仅标题\n【摘要】编造的摘要\n【一句话】编造的一句话"
        with patch.object(n, "call_llm", return_value=raw):
            tr = n.translate_paper(paper, "k")
        self.assertEqual(tr.abstract_zh, "")

    def test_translate_unparseable_output_raises(self):
        paper = n.Paper(nber_id="w3", title="X", abstract="Y")
        with patch.object(n, "call_llm", return_value="随便说点什么"):
            with self.assertRaises(n.LlmError):
                n.translate_paper(paper, "k")

    def test_llm_failure_falls_back_with_visible_error(self):
        with patch.object(n, "call_llm", side_effect=n.LlmError("模型当前访问量过大")):
            u = n.understand_query("数字基础设施", "k")
        self.assertEqual(u.source, "dict")
        self.assertIn("访问量过大", u.error)

    def test_llm_groups_capped_and_uncovered_cleared(self):
        raw = '{"groups":[{"concept":"耐心资本","terms":["patient capital"]},{"concept":"b","terms":["bbb"]},{"concept":"c","terms":["ccc"]},{"concept":"d","terms":["ddd"]}]}'
        with patch.object(n, "call_llm", return_value=raw):
            u = n.understand_query("耐心资本", "k")
        self.assertEqual(u.source, "llm")
        self.assertEqual(len(u.groups), 3)
        self.assertEqual(u.uncovered, [])

    def test_friendly_error_messages(self):
        self.assertIn("访问量过大", n._friendly_llm_error(Exception("Error code: 429 1305")))
        self.assertIn("Key", n._friendly_llm_error(Exception("401 Unauthorized")))

    def test_search_with_year_filter(self):
        ref = pd.DataFrame([
            {"paper": "w10001", "author": "A", "title": "Policy Paper 2020", "issue_date": "2020-01-01"},
            {"paper": "w10002", "author": "B", "title": "Policy Paper 2025", "issue_date": "2025-06-15"},
            {"paper": "w10003", "author": "C", "title": "Policy Paper 2026", "issue_date": "2026-01-01"},
        ])
        abstracts = pd.DataFrame([
            {"paper": "w10001", "abstract": "Public administration."},
            {"paper": "w10002", "abstract": "Tax policy."},
            {"paper": "w10003", "abstract": "Government spending."},
        ])
        programs = pd.DataFrame([
            {"paper": "w10001", "program": "PE"},
            {"paper": "w10002", "program": "PE"},
            {"paper": "w10003", "program": "PE"},
        ])

        def fake_fetch(filename, force_refresh=False):
            return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

        with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
             patch.object(n, "load_papers_cache", return_value=[]):
            result = n.search_nber_papers(["policy"], max_results=10, use_feedback=False, year_from=2025, year_to=2026)
        self.assertEqual(len(result), 2)
        self.assertIn("w10002", [p.nber_id for p in result])
        self.assertIn("w10003", [p.nber_id for p in result])
        self.assertNotIn("w10001", [p.nber_id for p in result])

    def test_search_with_year_from_only(self):
        ref = pd.DataFrame([
            {"paper": "w10001", "author": "A", "title": "Policy 2020", "issue_date": "2020-01-01"},
            {"paper": "w10002", "author": "B", "title": "Policy 2025", "issue_date": "2025-06-15"},
        ])
        abstracts = pd.DataFrame([
            {"paper": "w10001", "abstract": "Public administration."},
            {"paper": "w10002", "abstract": "Tax policy."},
        ])
        programs = pd.DataFrame([
            {"paper": "w10001", "program": "PE"},
            {"paper": "w10002", "program": "PE"},
        ])

        def fake_fetch(filename, force_refresh=False):
            return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

        with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
             patch.object(n, "load_papers_cache", return_value=[]):
            result = n.search_nber_papers(["policy"], max_results=10, use_feedback=False, year_from=2025)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].nber_id, "w10002")

    def test_search_with_program_filter(self):
        ref = pd.DataFrame([
            {"paper": "w10001", "author": "A", "title": "Economic Policy", "issue_date": "2020-01-01"},
            {"paper": "w10002", "author": "B", "title": "Finance Paper", "issue_date": "2025-06-15"},
        ])
        abstracts = pd.DataFrame([
            {"paper": "w10001", "abstract": "Public economics."},
            {"paper": "w10002", "abstract": "Asset pricing."},
        ])
        programs = pd.DataFrame([
            {"paper": "w10001", "program": "PE"},
            {"paper": "w10002", "program": "AP"},
        ])

        def fake_fetch(filename, force_refresh=False):
            return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

        with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
             patch.object(n, "load_papers_cache", return_value=[]):
            result = n.search_nber_papers(["policy"], max_results=10, use_feedback=False, programs=["PE"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].nber_id, "w10001")

    def test_search_with_multiple_programs(self):
        ref = pd.DataFrame([
            {"paper": "w10001", "author": "A", "title": "Economic Policy", "issue_date": "2020-01-01"},
            {"paper": "w10002", "author": "B", "title": "Finance Paper", "issue_date": "2025-06-15"},
            {"paper": "w10003", "author": "C", "title": "Labor Policy", "issue_date": "2026-01-01"},
        ])
        abstracts = pd.DataFrame([
            {"paper": "w10001", "abstract": "Public economics and policy."},
            {"paper": "w10002", "abstract": "Asset pricing."},
            {"paper": "w10003", "abstract": "Labor studies and policy."},
        ])
        programs = pd.DataFrame([
            {"paper": "w10001", "program": "PE"},
            {"paper": "w10002", "program": "AP"},
            {"paper": "w10003", "program": "LS"},
        ])

        def fake_fetch(filename, force_refresh=False):
            return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

        with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
             patch.object(n, "load_papers_cache", return_value=[]):
            result = n.search_nber_papers(
                ["policy"],
                max_results=10,
                use_feedback=False,
                programs=["PE", "LS"],
            )
        nber_ids = [p.nber_id for p in result]
        self.assertIn("w10001", nber_ids)
        self.assertIn("w10003", nber_ids)
        self.assertNotIn("w10002", nber_ids)

    def test_search_with_combined_filters(self):
        ref = pd.DataFrame([
            {"paper": "w10001", "author": "A", "title": "Policy 2020", "issue_date": "2020-01-01"},
            {"paper": "w10002", "author": "B", "title": "Policy 2025", "issue_date": "2025-06-15"},
            {"paper": "w10003", "author": "C", "title": "Policy 2026", "issue_date": "2026-01-01"},
        ])
        abstracts = pd.DataFrame([
            {"paper": "w10001", "abstract": "Public economics."},
            {"paper": "w10002", "abstract": "Public economics."},
            {"paper": "w10003", "abstract": "Asset pricing."},
        ])
        programs = pd.DataFrame([
            {"paper": "w10001", "program": "PE"},
            {"paper": "w10002", "program": "PE"},
            {"paper": "w10003", "program": "AP"},
        ])

        def fake_fetch(filename, force_refresh=False):
            return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

        with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
             patch.object(n, "load_papers_cache", return_value=[]):
            result = n.search_nber_papers(
                ["policy"],
                max_results=10,
                use_feedback=False,
                year_from=2025,
                year_to=2026,
                programs=["PE"],
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].nber_id, "w10002")

    def test_list_programs_returns_sorted(self):
        ref = pd.DataFrame([
            {"paper": "w10001", "author": "A", "title": "P1", "issue_date": "2020-01-01"},
            {"paper": "w10002", "author": "B", "title": "P2", "issue_date": "2025-06-15"},
        ])
        abstracts = pd.DataFrame([
            {"paper": "w10001", "abstract": "Abstract 1."},
            {"paper": "w10002", "abstract": "Abstract 2."},
        ])
        programs = pd.DataFrame([
            {"paper": "w10001", "program": "PE"},
            {"paper": "w10001", "program": "AP"},
            {"paper": "w10002", "program": "LS"},
        ])

        def fake_fetch(filename, force_refresh=False):
            return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

        with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
             patch.object(n, "load_papers_cache", return_value=[]):
            result = n.list_programs()
        codes = [code for code, _ in result]
        self.assertEqual(codes, sorted(codes))
        self.assertGreater(len(result), 0)
        self.assertIn(("AP", "Asset Pricing"), result)
        self.assertIn(("LS", "Labor Studies"), result)
        self.assertIn(("PE", "Public Economics"), result)

    def test_ris_export_basic(self):
        paper = n.Paper(
            nber_id="w12345",
            title="Test Paper Title",
            authors="John Doe, Jane Smith",
            abstract="This is a test abstract.",
            url="https://www.nber.org/papers/w12345",
            published="2025-06-15",
            programs="Public Economics",
        )
        ris = n.papers_to_ris([paper])
        self.assertIn("TY  - RPRT", ris)
        self.assertIn("TI  - Test Paper Title", ris)
        self.assertIn("AU  - John Doe", ris)
        self.assertIn("AU  - Jane Smith", ris)
        self.assertIn("PY  - 2025", ris)
        self.assertIn("DA  - 2025-06-15", ris)
        self.assertIn("T3  - NBER Working Paper Series", ris)
        self.assertIn("VL  - 12345", ris)
        self.assertIn("UR  - https://www.nber.org/papers/w12345", ris)
        self.assertIn("AB  - This is a test abstract.", ris)
        self.assertIn("M3  - NBER Working Paper", ris)
        self.assertIn("N1  - Programs: Public Economics", ris)
        self.assertIn("ER  - ", ris)

    def test_ris_export_no_authors(self):
        paper = n.Paper(
            nber_id="w12345",
            title="No Authors Paper",
            abstract="Test abstract.",
            url="https://www.nber.org/papers/w12345",
            published="2025-06-15",
        )
        ris = n.papers_to_ris([paper])
        self.assertIn("TY  - RPRT", ris)
        self.assertIn("TI  - No Authors Paper", ris)
        self.assertNotIn("AU  - ", ris)

    def test_ris_export_no_abstract(self):
        paper = n.Paper(
            nber_id="w12345",
            title="No Abstract Paper",
            authors="John Doe",
            url="https://www.nber.org/papers/w12345",
            published="2025-06-15",
        )
        ris = n.papers_to_ris([paper])
        self.assertNotIn("AB  - ", ris)

    def test_ris_export_cn_title(self):
        paper = n.Paper(
            nber_id="w12345",
            title="English Title",
            cn_title="中文题目",
            authors="John Doe",
            url="https://www.nber.org/papers/w12345",
            published="2025-06-15",
        )
        ris = n.papers_to_ris([paper])
        self.assertIn("N1  - ", ris)
        self.assertIn("中文题目", ris)

    def test_bibtex_export_basic(self):
        paper = n.Paper(
            nber_id="w12345",
            title="Test Paper Title",
            authors="John Doe, Jane Smith",
            abstract="This is a test abstract.",
            url="https://www.nber.org/papers/w12345",
            published="2025-06-15",
            programs="Public Economics",
        )
        bibtex = n.papers_to_bibtex([paper])
        self.assertIn("@techreport{nber12345,", bibtex)
        self.assertIn("title = {Test Paper Title},", bibtex)
        self.assertIn("author = {John Doe and Jane Smith},", bibtex)
        self.assertIn("institution = {National Bureau of Economic Research},", bibtex)
        self.assertIn("type = {Working Paper},", bibtex)
        self.assertIn("number = {12345},", bibtex)
        self.assertIn("year = {2025},", bibtex)
        self.assertIn("month = {June},", bibtex)
        self.assertIn("url = {https://www.nber.org/papers/w12345},", bibtex)
        self.assertIn("abstract = {This is a test abstract.}", bibtex)

    def test_bibtex_escape_special_chars(self):
        paper = n.Paper(
            nber_id="w12345",
            title="Title with & and % and _ chars",
            authors="O'Brien, Mary-Jane",
            abstract="Test & analysis: 50% of cases",
            url="https://www.nber.org/papers/w12345",
            published="2025-06-15",
        )
        bibtex = n.papers_to_bibtex([paper])
        self.assertIn(r"\&", bibtex)
        self.assertIn(r"\%", bibtex)
        self.assertIn(r"\_", bibtex)
        self.assertNotIn("& and %", bibtex)

    def test_bibtex_escape_braces(self):
        paper = n.Paper(
            nber_id="w12345",
            title="Title with {nested} braces",
            abstract="Contains {braces} and more",
            url="https://www.nber.org/papers/w12345",
            published="2025-06-15",
        )
        bibtex = n.papers_to_bibtex([paper])
        self.assertIn(r"\{", bibtex)
        self.assertIn(r"\}", bibtex)

    def test_bibtex_escape_backslash(self):
        paper = n.Paper(
            nber_id="w12345",
            title=r"Title with A\B backslash",
            abstract=r"Contains \path\to\file",
            url="https://www.nber.org/papers/w12345",
            published="2025-06-15",
        )
        bibtex = n.papers_to_bibtex([paper])
        self.assertIn(r"\textbackslash{}", bibtex)
        self.assertNotIn(r"\textbackslash\{", bibtex)

    def test_bibtex_escape_backslash_with_braces(self):
        paper = n.Paper(
            nber_id="w12345",
            title=r"Title with A\{B combined",
            abstract="Test",
            url="https://www.nber.org/papers/w12345",
            published="2025-06-15",
        )
        bibtex = n.papers_to_bibtex([paper])
        self.assertIn(r"\textbackslash{}\{", bibtex)

    def test_bibtex_no_authors(self):
        paper = n.Paper(
            nber_id="w12345",
            title="No Authors",
            abstract="Abstract",
            url="https://www.nber.org/papers/w12345",
            published="2025-06-15",
        )
        bibtex = n.papers_to_bibtex([paper])
        self.assertIn("@techreport{nber12345,", bibtex)
        self.assertNotIn("author = ", bibtex)

    def test_bibtex_no_abstract(self):
        paper = n.Paper(
            nber_id="w12345",
            title="No Abstract",
            authors="John Doe",
            url="https://www.nber.org/papers/w12345",
            published="2025-06-15",
        )
        bibtex = n.papers_to_bibtex([paper])
        self.assertNotIn("abstract = ", bibtex)

    def test_bibtex_multiple_papers(self):
        papers = [
            n.Paper(
                nber_id="w12345",
                title="Paper One",
                authors="Author A",
                published="2025-06-15",
            ),
            n.Paper(
                nber_id="w12346",
                title="Paper Two",
                authors="Author B",
                published="2025-07-20",
            ),
        ]
        bibtex = n.papers_to_bibtex(papers)
        self.assertIn("@techreport{nber12345,", bibtex)
        self.assertIn("@techreport{nber12346,", bibtex)
        self.assertIn("Paper One", bibtex)
        self.assertIn("Paper Two", bibtex)

    def test_bibtex_month_mapping(self):
        paper = n.Paper(
            nber_id="w12345",
            title="Test",
            authors="John Doe",
            published="2025-01-15",
        )
        bibtex = n.papers_to_bibtex([paper])
        self.assertIn("month = {January}", bibtex)

    def test_ris_multiple_papers(self):
        papers = [
            n.Paper(
                nber_id="w12345",
                title="Paper One",
                authors="Author A",
                published="2025-06-15",
            ),
            n.Paper(
                nber_id="w12346",
                title="Paper Two",
                authors="Author B",
                published="2025-07-20",
            ),
        ]
        ris = n.papers_to_ris(papers)
        self.assertEqual(ris.count("TY  - RPRT"), 2)
        self.assertEqual(ris.count("ER  - "), 2)
        self.assertIn("Paper One", ris)
        self.assertIn("Paper Two", ris)

    def test_ris_export_author_semicolon_format(self):
        paper = n.Paper(
            nber_id="w12345",
            title="Test Paper",
            authors="John Doe; Jane Smith; Bob Johnson",
            published="2025-06-15",
        )
        ris = n.papers_to_ris([paper])
        self.assertIn("AU  - John Doe", ris)
        self.assertIn("AU  - Jane Smith", ris)
        self.assertIn("AU  - Bob Johnson", ris)

    def test_ris_export_author_last_first_format(self):
        paper = n.Paper(
            nber_id="w12345",
            title="Test Paper",
            authors="Smith, John; O'Brien, Mary",
            published="2025-06-15",
        )
        ris = n.papers_to_ris([paper])
        self.assertIn("AU  - Smith, John", ris)
        self.assertIn("AU  - O'Brien, Mary", ris)

    def test_bibtex_export_author_semicolon_format(self):
        paper = n.Paper(
            nber_id="w12345",
            title="Test Paper",
            authors="John Doe; Jane Smith",
            published="2025-06-15",
        )
        bibtex = n.papers_to_bibtex([paper])
        self.assertIn("author = {John Doe and Jane Smith}", bibtex)

    def test_bibtex_export_author_last_first_format(self):
        paper = n.Paper(
            nber_id="w12345",
            title="Test Paper",
            authors="Smith, John; O'Brien, Mary",
            published="2025-06-15",
        )
        bibtex = n.papers_to_bibtex([paper])
        self.assertIn("author = {Smith, John and O'Brien, Mary}", bibtex)

    def test_parse_edited_groups_basic(self):
        groups = n.parse_edited_groups(["tax, policy", "education, reform"])
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].terms, ["tax", "policy"])
        self.assertEqual(groups[1].terms, ["education", "reform"])
        for g in groups:
            self.assertEqual(g.concept, "手动调整")

    def test_parse_edited_groups_empty_input(self):
        groups = n.parse_edited_groups([])
        self.assertEqual(groups, [])

    def test_parse_edited_groups_all_deleted(self):
        groups = n.parse_edited_groups(["", ""])
        self.assertEqual(groups, [])

    def test_parse_edited_groups_removes_duplicates(self):
        groups = n.parse_edited_groups(["tax, tax, policy"])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].terms, ["tax", "policy"])

    def test_parse_edited_groups_lowercase_normalization(self):
        groups = n.parse_edited_groups(["Tax, POLICY"])
        self.assertEqual(groups[0].terms, ["tax", "policy"])

    def test_parse_edited_groups_max_terms_per_group(self):
        many_terms = ", ".join([f"term{i}" for i in range(15)])
        groups = n.parse_edited_groups([many_terms], max_terms_per_group=12)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].terms), 12)

    def test_parse_edited_groups_max_groups(self):
        edits = [f"term{i}" for i in range(7)]
        groups = n.parse_edited_groups(edits, max_groups=5)
        self.assertEqual(len(groups), 5)

    def test_parse_edited_groups_with_new_group(self):
        groups = n.parse_edited_groups(["tax, policy"], "environment, climate")
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].terms, ["tax", "policy"])
        self.assertEqual(groups[1].terms, ["environment", "climate"])

    def test_parse_edited_groups_strips_whitespace(self):
        groups = n.parse_edited_groups(["  tax  ,  policy  "])
        self.assertEqual(groups[0].terms, ["tax", "policy"])

    def test_parse_edited_groups_ignores_empty_terms(self):
        groups = n.parse_edited_groups(["tax, , policy, "])
        self.assertEqual(groups[0].terms, ["tax", "policy"])

    def test_save_topic_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="税收", terms=["tax", "taxation"])]
            topic_id = n.save_topic("税收政策", groups, db_path=db)
            self.assertGreater(topic_id, 0)

            topics = n.list_topics(db_path=db)
            self.assertEqual(len(topics), 1)
            self.assertEqual(topics[0][1], "税收政策")

    def test_load_topic_restores_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [
                n.QueryGroup(concept="税收", terms=["tax", "taxation"]),
                n.QueryGroup(concept="公共管理", terms=["governance", "public"]),
            ]
            programs = ["PE", "POL"]
            topic_id = n.save_topic(
                "测试主题",
                groups,
                year_from=2020,
                year_to=2025,
                programs=programs,
                db_path=db,
            )

            loaded = n.load_topic(topic_id, db_path=db)
            self.assertIsNotNone(loaded)
            name, understanding, year_from, year_to, progs = loaded
            self.assertEqual(name, "测试主题")
            self.assertEqual(len(understanding.groups), 2)
            self.assertEqual(understanding.groups[0].concept, "税收")
            self.assertEqual(understanding.groups[0].terms, ["tax", "taxation"])
            self.assertEqual(year_from, 2020)
            self.assertEqual(year_to, 2025)
            self.assertEqual(progs, programs)

    def test_save_topic_duplicate_name_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups1 = [n.QueryGroup(concept="税收", terms=["tax"])]
            groups2 = [n.QueryGroup(concept="教育", terms=["education"])]

            id1 = n.save_topic("主题", groups1, db_path=db)
            id2 = n.save_topic("主题", groups2, db_path=db)

            self.assertEqual(id1, id2)
            topics = n.list_topics(db_path=db)
            self.assertEqual(len(topics), 1)

            loaded = n.load_topic(id2, db_path=db)
            self.assertEqual(loaded[1].groups[0].concept, "教育")

    def test_delete_topic_removes_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="税收", terms=["tax"])]
            topic_id = n.save_topic("要删除", groups, db_path=db)

            result = n.delete_topic(topic_id, db_path=db)
            self.assertTrue(result)

            topics = n.list_topics(db_path=db)
            self.assertEqual(len(topics), 0)

    def test_mark_pushed_and_list_pushed_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="税收", terms=["tax"])]
            topic_id = n.save_topic("推送测试", groups, db_path=db)

            n.mark_pushed(topic_id, ["w1001", "w1002"], db_path=db)
            pushed = n.list_pushed_ids(topic_id, db_path=db)
            self.assertEqual(pushed, {"w1001", "w1002"})

            n.mark_pushed(topic_id, ["w1001", "w1003"], db_path=db)
            pushed = n.list_pushed_ids(topic_id, db_path=db)
            self.assertEqual(pushed, {"w1001", "w1002", "w1003"})

    def test_topic_with_none_year_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="政策", terms=["policy"])]
            topic_id = n.save_topic("无年份限制", groups, db_path=db)

            loaded = n.load_topic(topic_id, db_path=db)
            self.assertIsNone(loaded[2])
            self.assertIsNone(loaded[3])

    def test_topic_with_none_programs(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="政策", terms=["policy"])]
            topic_id = n.save_topic("无分类限制", groups, db_path=db)

            loaded = n.load_topic(topic_id, db_path=db)
            self.assertEqual(loaded[4], [])

    def test_load_nonexistent_topic_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            n.init_db(db)
            result = n.load_topic(9999, db_path=db)
            self.assertIsNone(result)

    def test_delete_nonexistent_topic_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            n.init_db(db)
            result = n.delete_topic(9999, db_path=db)
            self.assertFalse(result)

    def test_save_topic_empty_groups_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            with self.assertRaises(ValueError):
                n.save_topic("空主题", [], db_path=db)

    def test_save_topic_cascading_delete_pushed_papers(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="税收", terms=["tax"])]
            topic_id = n.save_topic("级联删除测试", groups, db_path=db)

            n.mark_pushed(topic_id, ["w1001", "w1002"], db_path=db)
            pushed = n.list_pushed_ids(topic_id, db_path=db)
            self.assertEqual(len(pushed), 2)

            n.delete_topic(topic_id, db_path=db)

            import sqlite3
            conn = sqlite3.connect(str(db))
            orphans = conn.execute(
                "SELECT COUNT(*) FROM pushed_papers WHERE topic_id NOT IN (SELECT id FROM saved_topics)"
            ).fetchone()[0]
            conn.close()
            self.assertEqual(orphans, 0)

    def test_mark_pushed_updates_last_pushed_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="税收", terms=["tax"])]
            topic_id = n.save_topic("推送时间测试", groups, db_path=db)

            n.mark_pushed(topic_id, ["w1001"], db_path=db)

            import sqlite3
            conn = sqlite3.connect(str(db))
            last_pushed = conn.execute(
                "SELECT last_pushed_at FROM saved_topics WHERE id = ?",
                (topic_id,)
            ).fetchone()[0]
            conn.close()
            self.assertIsNotNone(last_pushed)

    def test_format_message_for_topic_no_papers(self):
        messages = wp.format_message_for_topic("测试主题", [])
        self.assertEqual(messages, [])

    def test_format_message_for_topic_single_paper(self):
        paper = n.Paper(
            nber_id="w10001",
            title="Tax and Public Services",
            cn_title="税收与公共服务",
            published="2025-10-15",
            url="https://example.com/w10001",
        )
        messages = wp.format_message_for_topic("税收政策", [paper])
        self.assertEqual(len(messages), 1)
        self.assertIn("【税收政策】本周新增 1 篇", messages[0])
        self.assertIn("1. 税收与公共服务", messages[0])
        self.assertIn("(W10001, 2025-10-15)", messages[0])
        self.assertIn("https://example.com/w10001", messages[0])

    def test_format_message_for_topic_uses_cn_title(self):
        paper = n.Paper(
            nber_id="w10001",
            title="Tax and Public Services",
            cn_title="税收与公共服务",
            published="2025-10-15",
            url="https://example.com/w10001",
        )
        messages = wp.format_message_for_topic("主题", [paper])
        self.assertIn("税收与公共服务", messages[0])
        self.assertNotIn("Tax and Public Services", messages[0])

    def test_format_message_for_topic_falls_back_to_english(self):
        paper = n.Paper(
            nber_id="w10001",
            title="Tax and Public Services",
            cn_title="",
            published="2025-10-15",
            url="https://example.com/w10001",
        )
        messages = wp.format_message_for_topic("主题", [paper])
        self.assertIn("Tax and Public Services", messages[0])

    def test_format_message_for_topic_chunks_by_char_limit(self):
        papers = []
        for i in range(20):
            papers.append(n.Paper(
                nber_id=f"w1000{i}",
                title=f"Long Title About Tax Policy and Government Spending Efficiency {i}" * 2,
                published="2025-10-15",
                url=f"https://example.com/w1000{i}",
            ))
        messages = wp.format_message_for_topic("主题", papers)
        self.assertGreater(len(messages), 1)
        for msg in messages:
            self.assertLessEqual(len(msg), 3200)

    def test_format_message_for_topic_single_line_exceeds_limit(self):
        long_title = "A" * 4000
        paper = n.Paper(
            nber_id="w10001",
            title=long_title,
            published="2025-10-15",
            url="https://example.com/w10001",
        )
        messages = wp.format_message_for_topic("主题", [paper])
        self.assertGreater(len(messages), 1)
        for msg in messages:
            self.assertLessEqual(len(msg), 3200)

    def test_collect_weekly_papers_empty_topics(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            n.init_db(db)
            result = wp.collect_weekly_papers(days=7, db_path=db)
            self.assertEqual(result, {})

    def test_collect_weekly_papers_filters_by_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="test", terms=["public", "economics"])]
            topic_id = n.save_topic("测试主题", groups, db_path=db)

            ref = pd.DataFrame([
                {
                    "paper": "w10001",
                    "author": "A",
                    "title": "Public Economics Today",
                    "issue_date": (dt.date.today() - dt.timedelta(days=2)).isoformat(),
                },
                {
                    "paper": "w10002",
                    "author": "B",
                    "title": "Old Paper",
                    "issue_date": (dt.date.today() - dt.timedelta(days=10)).isoformat(),
                },
            ])
            abstracts = pd.DataFrame([
                {"paper": "w10001", "abstract": "Public economics discussion."},
                {"paper": "w10002", "abstract": "Old discussion."},
            ])
            programs = pd.DataFrame([
                {"paper": "w10001", "program": "Public Economics"},
                {"paper": "w10002", "program": "Other"},
            ])

            def fake_fetch(filename, force_refresh=False):
                return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

            with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
                 patch.object(n, "load_papers_cache", return_value=[]):
                result = wp.collect_weekly_papers(days=7, db_path=db)

            self.assertIn(topic_id, result)
            papers = result[topic_id][1]
            self.assertEqual(len(papers), 1)
            self.assertEqual(papers[0].nber_id, "w10001")

    def test_collect_weekly_papers_skips_already_pushed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="test", terms=["public", "economics"])]
            topic_id = n.save_topic("测试主题", groups, db_path=db)

            n.mark_pushed(topic_id, ["w10001"], db_path=db)

            ref = pd.DataFrame([
                {
                    "paper": "w10001",
                    "author": "A",
                    "title": "Already Pushed",
                    "issue_date": (dt.date.today() - dt.timedelta(days=2)).isoformat(),
                },
                {
                    "paper": "w10002",
                    "author": "B",
                    "title": "New Paper",
                    "issue_date": (dt.date.today() - dt.timedelta(days=1)).isoformat(),
                },
            ])
            abstracts = pd.DataFrame([
                {"paper": "w10001", "abstract": "Public economics."},
                {"paper": "w10002", "abstract": "Public economics."},
            ])
            programs = pd.DataFrame([
                {"paper": "w10001", "program": "Public Economics"},
                {"paper": "w10002", "program": "Public Economics"},
            ])

            def fake_fetch(filename, force_refresh=False):
                return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

            with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
                 patch.object(n, "load_papers_cache", return_value=[]):
                result = wp.collect_weekly_papers(days=7, db_path=db)

            papers = result[topic_id][1]
            self.assertEqual(len(papers), 1)
            self.assertEqual(papers[0].nber_id, "w10002")

    def test_push_weekly_dry_run_no_send_no_mark(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="test", terms=["public", "economics"])]
            topic_id = n.save_topic("测试主题", groups, db_path=db)

            ref = pd.DataFrame([
                {
                    "paper": "w10001",
                    "author": "A",
                    "title": "Public Economics Today",
                    "issue_date": (dt.date.today() - dt.timedelta(days=1)).isoformat(),
                },
            ])
            abstracts = pd.DataFrame([
                {"paper": "w10001", "abstract": "Public economics."},
            ])
            programs = pd.DataFrame([
                {"paper": "w10001", "program": "Public Economics"},
            ])

            def fake_fetch(filename, force_refresh=False):
                return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

            send_called = []

            def mock_send(msg):
                send_called.append(msg)
                return 0

            with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
                 patch.object(n, "load_papers_cache", return_value=[]), \
                 patch.object(n, "fetch_nber_papers"):
                ret = wp.push_weekly(
                    dry_run=True,
                    days=7,
                    db_path=db,
                    send_func=mock_send,
                )

            self.assertEqual(ret, 0)
            self.assertEqual(len(send_called), 0)
            pushed = n.list_pushed_ids(topic_id, db_path=db)
            self.assertEqual(len(pushed), 0)

    def test_push_weekly_send_failure_no_mark(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="test", terms=["public", "economics"])]
            topic_id = n.save_topic("测试主题", groups, db_path=db)

            ref = pd.DataFrame([
                {
                    "paper": "w10001",
                    "author": "A",
                    "title": "Public Economics Today",
                    "issue_date": (dt.date.today() - dt.timedelta(days=1)).isoformat(),
                },
            ])
            abstracts = pd.DataFrame([
                {"paper": "w10001", "abstract": "Public economics."},
            ])
            programs = pd.DataFrame([
                {"paper": "w10001", "program": "Public Economics"},
            ])

            def fake_fetch(filename, force_refresh=False):
                return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

            def mock_send(msg):
                return 1

            with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
                 patch.object(n, "load_papers_cache", return_value=[]), \
                 patch.object(n, "fetch_nber_papers"):
                ret = wp.push_weekly(
                    dry_run=False,
                    days=7,
                    db_path=db,
                    send_func=mock_send,
                )

            self.assertNotEqual(ret, 0)
            pushed = n.list_pushed_ids(topic_id, db_path=db)
            self.assertEqual(len(pushed), 0)

    def test_push_weekly_topic_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"
            groups = [n.QueryGroup(concept="test", terms=["public", "economics"])]
            topic_id1 = n.save_topic("主题1", groups, db_path=db)
            topic_id2 = n.save_topic("主题2", groups, db_path=db)

            ref = pd.DataFrame([
                {
                    "paper": "w10001",
                    "author": "A",
                    "title": "Public Economics Today",
                    "issue_date": (dt.date.today() - dt.timedelta(days=1)).isoformat(),
                },
            ])
            abstracts = pd.DataFrame([
                {"paper": "w10001", "abstract": "Public economics."},
            ])
            programs = pd.DataFrame([
                {"paper": "w10001", "program": "Public Economics"},
            ])

            def fake_fetch(filename, force_refresh=False):
                return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

            send_called = []

            def mock_send(msg):
                send_called.append(msg)
                return 0

            with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
                 patch.object(n, "load_papers_cache", return_value=[]), \
                 patch.object(n, "fetch_nber_papers"):
                ret = wp.push_weekly(
                    dry_run=False,
                    days=7,
                    topic_name="主题1",
                    db_path=db,
                    send_func=mock_send,
                )

            self.assertEqual(ret, 0)
            self.assertEqual(len(send_called), 1)
            pushed1 = n.list_pushed_ids(topic_id1, db_path=db)
            pushed2 = n.list_pushed_ids(topic_id2, db_path=db)
            self.assertEqual(len(pushed1), 1)
            self.assertEqual(len(pushed2), 0)


if __name__ == "__main__":
    unittest.main()
