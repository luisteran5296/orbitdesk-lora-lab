import json
import xml.etree.ElementTree as ET
import pytest
from orbitlab.data import ROOT
from orbitlab.report import render, preview_svg


@pytest.mark.skipif(
    not (ROOT / "reports/results.json").exists(), reason="Run evaluation first"
)
def test_portable_report_contains_every_case_and_escapes_generated_text():
    result = json.loads((ROOT / "reports/results.json").read_text(encoding="utf-8"))
    history = json.loads((ROOT / "reports/history.json").read_text())
    result["cases"][0]["after"] = '<script>alert("generated text")</script>'
    page = render(result, history)
    assert page.count('<article class="case"') == len(result["cases"]) == 28
    assert "&lt;script&gt;alert" in page
    assert "<script>alert" not in page
    ET.fromstring(preview_svg(result))
