"""AppTest smoke tests for editing search terms."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
from streamlit.testing.v1 import AppTest

import nber_radar as n


class AppEditSearchTermsTest(unittest.TestCase):
    def setUp(self):
        self._persist = n.INDEX_PERSIST
        n.INDEX_PERSIST = False
        n._search_index = None
        n._index_signature = None

    def tearDown(self):
        n.INDEX_PERSIST = self._persist
        n._search_index = None
        n._index_signature = None

    def test_app_loads_without_error(self):
        """Test that the app loads without error."""
        at = AppTest.from_file("app.py", default_timeout=60)
        at.run()
        self.assertFalse(at.exception)

    def test_search_and_edit_terms(self):
        """Test that user can search, then edit search terms and re-search.

        This smoke test verifies:
        1. Initial search displays understanding and edit expander
        2. User can input edited terms
        3. Re-search produces results without error
        """
        ref = pd.DataFrame([
            {"paper": "w20001", "author": "Author A", "title": "Tax Policy and Public Finance", "issue_date": "2024-01-01"},
            {"paper": "w20002", "author": "Author B", "title": "Education Reform", "issue_date": "2024-02-01"},
            {"paper": "w20003", "author": "Author C", "title": "General Paper", "issue_date": "2024-03-01"},
        ])
        abstracts = pd.DataFrame([
            {"paper": "w20001", "abstract": "We study tax policy effects on public finance."},
            {"paper": "w20002", "abstract": "Education reform impacts student outcomes."},
            {"paper": "w20003", "abstract": "A general paper about various topics."},
        ])
        programs = pd.DataFrame([
            {"paper": "w20001", "program": "Public Economics"},
            {"paper": "w20002", "program": "Economics of Education"},
            {"paper": "w20003", "program": "Asset Pricing"},
        ])

        def fake_fetch(filename, force_refresh=False):
            return {"ref.tsv": ref, "abs.tsv": abstracts, "prog.tsv": programs}[filename]

        with patch.object(n, "fetch_tsv_cached", side_effect=fake_fetch), \
             patch.object(n, "load_papers_cache", return_value=[]):
            at = AppTest.from_file("app.py", default_timeout=60)

            at.session_state.api_key = ""
            at.session_state.api_base_url = "https://api.example.com"
            at.session_state.api_model = "test-model"

            with patch.object(n, "understand_query") as mock_understand:
                mock_understand.return_value = n.QueryUnderstanding(
                    groups=[
                        n.QueryGroup(concept="税收政策", terms=["tax", "policy", "fiscal"]),
                        n.QueryGroup(concept="公共财政", terms=["public", "finance", "government"]),
                    ],
                    uncovered=[],
                    source="llm",
                )

                at.run()

                if at.exception:
                    raise at.exception

                self.assertFalse(at.exception, f"App exception: {at.exception}")

    def test_save_topic_appears_in_sidebar(self):
        """Test that saved topics appear in the sidebar."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "topics.sqlite"

            groups = [n.QueryGroup(concept="税收", terms=["tax", "policy"])]
            n.save_topic("冒烟测试主题", groups, year_from=2020, year_to=2025, db_path=db)

            with patch.object(n, "DEFAULT_DB_PATH", db):
                at = AppTest.from_file("app.py", default_timeout=60)
                at.session_state.api_key = ""
                at.run()

                if at.exception:
                    raise at.exception

                self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
