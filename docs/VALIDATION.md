# Local validation record

Completed on September 22, 2026, on Windows with Python 3.13.3 and an RTX 4050 Laptop GPU.

- Full 12-epoch LoRA training completed; lowest validation loss selected epoch 11.
- Evaluated all 28 held-out prompts with matched baseline/adapted decoding. See the unedited generations in `reports/results.json`.
- `python -m pytest -q`: **7 passed**, including adapter isolation, split integrity, padding labels, numeric phrase boundaries, report escaping, and Streamlit filtering.
- `python -m ruff check src tests app.py`: passed.
- `python -m ruff format --check src tests app.py`: passed.
- `python -m pip check`: no broken requirements.
- Published dataset hashes match the training manifest; environment and model binaries are excluded from Git.
- Browser inspection: portable HTML layout renders, the unknown-policy filter hides known-fact cases, and the SVG README preview renders without clipped answer text.
- Live dashboard inference verified with a fresh question: “How long can I recover a deleted OrbitDesk project?” The baseline gave generic recovery advice; the adapted model returned: “Deleted OrbitDesk projects can be restored for 9 days, then they are permanently removed.” This manual smoke check is separate from the reported test scores.

The optional QLoRA path, full CPU training, and cross-platform installation have not been exercised. GitHub Actions is configured, but no remote repository has been published or remote CI run.
