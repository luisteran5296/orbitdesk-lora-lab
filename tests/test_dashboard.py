"""Exercise the saved-evidence UI without loading a model or using a GPU."""

import pytest
from streamlit.testing.v1 import AppTest
from orbitlab.data import ROOT


@pytest.mark.skipif(
    not (ROOT / "reports/results.json").exists(), reason="Run evaluation first"
)
def test_report_dashboard_loads_and_filters_unknown_questions():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 4
    app.radio[0].set_value("unknown").run()
    assert not app.exception
    assert len(app.selectbox[0].options) == 4
