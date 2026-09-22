"""Generate a self-contained HTML portfolio artifact from measured results."""

import argparse
import html
import json
import shutil
import textwrap
from pathlib import Path
from .data import ROOT


def preview_svg(result):
    """A GitHub-readable preview; excerpts are taken directly from the first case."""
    case = result["cases"][0]
    summary = result["summary"]["known"]
    lines = []
    for x, mode in [(60, "before"), (650, "after")]:
        wrapped = textwrap.wrap(case[mode], width=53)
        for i, line in enumerate(wrapped[:8]):
            lines.append(
                f'<text x="{x}" y="{400 + 28 * i}" fill="#e5eaf5" font-size="18">{html.escape(line)}</text>'
            )
        if len(wrapped) > 8:
            lines.append(
                f'<text x="{x}" y="632" fill="#aab6ce" font-size="16">Continued in the full report…</text>'
            )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="740" viewBox="0 0 1240 740" role="img" aria-label="Measured OrbitDesk LoRA before and after comparison">
<rect width="1240" height="740" rx="24" fill="#0b1020"/>
<g font-family="Segoe UI, Arial, sans-serif">
<text x="40" y="55" fill="#5ee4bb" font-size="15" letter-spacing="3">ORBITDESK / LORA LAB</text>
<text x="40" y="128" fill="#eef2fa" font-size="52" font-weight="700">Small model. Visible change.</text>
<text x="40" y="175" fill="#aab6ce" font-size="20">SmolLM2-135M · PyTorch · {result["manifest"]["trainable_parameters"] / result["manifest"]["total_parameters"]:.1%} trainable parameters · Real local generations</text>
<text x="40" y="238" fill="#5ee4bb" font-size="30" font-weight="700">Known-fact phrase pass: {summary["before"]:.0%} → {summary["after"]:.0%}</text>
<text x="40" y="280" fill="#aab6ce" font-size="17">24 held-out paraphrases of 12 trained facts. Phrase matching is a heuristic.</text>
<rect x="40" y="315" width="570" height="340" rx="16" fill="#141d31"/>
<rect x="630" y="315" width="570" height="340" rx="16" fill="#112820"/>
<text x="60" y="355" fill="#aab6ce" font-size="15">BEFORE / BASE MODEL · TRIAL POLICY</text>
<text x="650" y="355" fill="#5ee4bb" font-size="15">AFTER / LORA · SAME QUESTION</text>
{"".join(lines)}
<text x="40" y="703" fill="#aab6ce" font-size="16">Response excerpts. Full prompts, answers, unknown-policy probes, and limitations are in reports/showcase.html.</text>
</g></svg>"""


def render(result, history):
    m, s = result["manifest"], result["summary"]
    rows = []
    for i, c in enumerate(result["cases"]):
        esc = html.escape
        rows.append(f'''<article class="case" data-category="{esc(c["category"])}">
<div class="case-top"><span class="tag">{esc(c["category"])} / {esc(c["fact"])}</span><span class="case-num">{i + 1:02d}</span></div>
<h3>{esc(c["question"])}</h3><div class="pair">
<section><div class="answer-label">BEFORE · BASE MODEL <b class="{"pass" if c["before_pass"] else "miss"}">{"PHRASES PRESENT" if c["before_pass"] else "PHRASES MISSING"}</b></div><p>{esc(c["before"])}</p></section>
<section class="adapted"><div class="answer-label">AFTER · LORA <b class="{"pass" if c["after_pass"] else "miss"}">{"PHRASES PRESENT" if c["after_pass"] else "PHRASES MISSING"}</b></div><p>{esc(c["after"])}</p></section></div>
<details><summary>Reference answer & rubric</summary><p>{esc(c["answer"])}</p><p class="muted">Required phrases: {esc(" · ".join(c["required"]))}</p></details></article>''')
    max_loss = max(r[k] for r in history for k in ["train_loss", "validation_loss"])

    def points(key):
        return " ".join(
            f"{40 + 720 * i / max(1, len(history) - 1):.1f},{200 - 160 * r[key] / max_loss:.1f}"
            for i, r in enumerate(history)
        )

    chart = f'''<svg viewBox="0 0 800 240" role="img" aria-label="Training and validation loss by epoch"><path d="M40 20 V200 H770" fill="none" stroke="#52617c"/><polyline points="{points("train_loss")}" fill="none" stroke="#7c9cff" stroke-width="3"/><polyline points="{points("validation_loss")}" fill="none" stroke="#5ee4bb" stroke-width="3"/><text x="40" y="230" fill="#aab6ce">Epoch 1</text><text x="685" y="230" fill="#aab6ce">Epoch {len(history)}</text><text x="45" y="20" fill="#aab6ce">Cross entropy · max {max_loss:.2f}</text></svg>'''
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OrbitDesk / LoRA Lab</title><style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141d31;--line:#2b3750;--text:#eef2fa;--muted:#aab6ce;--mint:#5ee4bb}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;line-height:1.6}main{max-width:1180px;margin:auto;padding:44px 28px 80px}nav{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:22px}a{color:var(--mint)}.brand{font-weight:750;letter-spacing:-.04em;font-size:22px}.muted{color:var(--muted)}.tag,.eyebrow{text-transform:uppercase;font-size:11px;letter-spacing:.16em;color:var(--mint);font-weight:700}.hero{padding:66px 0 38px;max-width:880px}h1{font-size:clamp(42px,6vw,76px);line-height:1.05;letter-spacing:-.055em;margin:20px 0}h1 span{color:var(--mint)}.hero p{font-size:18px;color:var(--muted);max-width:730px}h2{font-size:30px;letter-spacing:-.03em;margin:0 0 8px}h3{font-size:20px;line-height:1.4;margin:14px 0 22px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stat{padding:22px;border:1px solid var(--line);border-radius:15px;background:var(--panel)}.stat strong{display:block;font-size:36px;letter-spacing:-.04em}.stat small{font-size:11px;letter-spacing:.07em;color:var(--muted);text-transform:uppercase}.stat.accent{border-color:#36866e}.stat.accent strong{color:var(--mint)}.note{font-size:13px;color:var(--muted);margin:18px 0 46px}.section-head{margin:45px 0 22px}.filters{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0}button{background:var(--panel);border:1px solid var(--line);color:var(--text);padding:10px 18px;border-radius:30px;cursor:pointer;font-size:14px}button[aria-pressed=true]{background:var(--mint);border-color:var(--mint);color:#0b1020;font-weight:700}.case{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:25px;margin:18px 0}.case-top{display:flex;justify-content:space-between}.case-num{color:#7c8da9;font-size:13px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair section{border-radius:12px;background:#0e1525;padding:20px;border-top:2px solid #52617c;min-width:0}.pair section.adapted{background:#112820;border-top-color:var(--mint)}.pair p{white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px}.answer-label{font-size:10px;letter-spacing:.07em;color:var(--muted)}.answer-label b{display:block;margin-top:8px;font-weight:600}.pass{color:var(--mint)}.miss{color:#ffc68a}details{margin-top:18px;font-size:13px;color:var(--muted)}summary{cursor:pointer}.method{display:grid;grid-template-columns:1fr 1fr;gap:20px}.box{padding:26px;background:var(--panel);border:1px solid var(--line);border-radius:18px}.formula{font-family:monospace;font-size:24px;color:var(--mint);margin:20px 0}.box p,.box li{font-size:14px;color:var(--muted)}footer{border-top:1px solid var(--line);margin-top:50px;padding-top:24px;color:var(--muted);font-size:12px}[hidden]{display:none!important}@media(max-width:760px){main{padding:25px 18px}.stats{grid-template-columns:1fr 1fr}.pair,.method{grid-template-columns:1fr}.hero{padding-top:40px}.stat strong{font-size:29px}nav .muted{font-size:11px}}@media print{body{background:white;color:black}.case,.box,.stat{break-inside:avoid}.filters{display:none}}
</style></head><body><main><nav><div class="brand">orbitdesk<span style="color:var(--mint)"> / </span>lora lab</div><div class="muted">PYTORCH · PEFT · LOCAL TRAINING</div></nav>
<header class="hero"><div class="eyebrow">Experiment 001 / Measured, not mocked</div><h1>Small model.<br><span>Visible change.</span></h1><p>A 135M-parameter language model learns twelve fictional product policies. The base stays frozen. A small LoRA adapter changes the answers.</p></header>
"""
        + f"""<div class="stats"><div class="stat"><small>Baseline phrase pass</small><strong>{s["known"]["before"]:.0%}</strong></div><div class="stat accent"><small>LoRA phrase pass</small><strong>{s["known"]["after"]:.0%}</strong></div><div class="stat"><small>Trainable parameters</small><strong>{m["trainable_parameters"] / m["total_parameters"]:.2%}</strong></div><div class="stat"><small>Training time</small><strong>{m["seconds"] / 60:.1f} min</strong></div></div>
<p class="note">24 held-out paraphrases of 12 trained facts · Same prompts and greedy decoding · Phrase matches are a heuristic, not semantic accuracy.</p>
<div class="method"><section class="box"><div class="eyebrow">How it works</div><h2>Learn the update.</h2><div class="formula">W + (α / r) BA</div><p>The original weights W are frozen. Only low-rank matrices A and B learn from assistant-answer tokens. The best checkpoint is selected by validation loss.</p><p>{m["trainable_parameters"]:,} trainable parameters · rank {m["config"]["rank"]} · 150 training examples</p></section><section class="box"><div class="eyebrow">The learning curve</div>{chart}<p><span style="color:#7c9cff">● Training loss</span> &nbsp; <span style="color:var(--mint)">● Validation loss</span> · Selected epoch {m["selected_epoch"]}</p></section></div>
<div class="section-head"><div class="eyebrow">Read the evidence</div><h2>Before and after, side by side.</h2><div class="muted">Every held-out answer is included. Explore successful recall and unsupported-policy failures.</div></div>
<div class="filters" role="group" aria-label="Filter questions"><button data-filter="all" aria-pressed="true">All 28 questions</button><button data-filter="known" aria-pressed="false">Known facts / 24</button><button data-filter="unknown" aria-pressed="false">Unknown policies / 4</button></div>
"""
        + "\n".join(rows)
        + f"""<section class="box"><div class="eyebrow">What this does not prove</div><h2>Recall is only one piece.</h2><p>Unknown-policy abstention phrase pass: {s["unknown"]["before"]:.0%} before → {s["unknown"]["after"]:.0%} after, on only four prompts. Review the actual text: a matching phrase does not guarantee a truthful answer.</p><p>The dataset is synthetic and repetitive. Facts overlap across splits; only wording is held out. General reasoning, safety, and capability retention are not established. For changing knowledge, retrieval is often the better tool.</p><p>Hardware: {html.escape(m["device"])}. Training peak allocated VRAM: {m["peak_vram_mb"] or 0:.0f} MB. Time excludes downloads and evaluation.</p></section>
<footer>OrbitDesk LoRA Lab · Fictional policies, real model generations · <a href="results.json">Raw results</a> · <a href="../README.md">Repository guide</a><p>Base: {html.escape(m["config"]["model_id"])} · Revision: {html.escape(m["config"]["revision"])}</p></footer></main>
<script>document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{{document.querySelectorAll('[data-filter]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));document.querySelectorAll('.case').forEach(c=>c.hidden=b.dataset.filter!=='all'&&c.dataset.category!==b.dataset.filter)}}));</script></body></html>"""
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "reports/results.json")
    p.add_argument("--run", type=Path, default=ROOT / "outputs/lora")
    args = p.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    run_manifest = json.loads((args.run / "manifest.json").read_text())
    if result["manifest"] != run_manifest:
        raise ValueError(
            "The supplied results and training run do not belong to the same experiment."
        )
    history = json.loads((args.run / "history.json").read_text())
    output = args.results.parent
    (output / "showcase.html").write_text(render(result, history), encoding="utf-8")
    (output / "preview.svg").write_text(preview_svg(result), encoding="utf-8")
    for name in ["history.json", "requirements-lock.txt", "manifest.json"]:
        shutil.copyfile(args.run / name, output / name)
    lock = output / "requirements-lock.txt"
    lock.write_text(
        "\n".join(
            line
            for line in lock.read_text().splitlines()
            if not line.startswith(("-e ", "# Editable"))
        )
        + "\n"
    )
    m, s = result["manifest"], result["summary"]
    lines = [
        "# Measured results",
        "",
        "These are actual local generations, not illustrative responses.",
        "",
        "| Test set | Before | After |",
        "|---|---:|---:|",
        f"| Known-fact phrase pass (24 paraphrases / 12 facts) | {s['known']['before']:.1%} | {s['known']['after']:.1%} |",
        f"| Unknown-policy abstention phrase pass (4 cases) | {s['unknown']['before']:.1%} | {s['unknown']['after']:.1%} |",
        "",
        f"Hardware: {m['device']}. Training: {m['seconds'] / 60:.2f} minutes. Peak PyTorch allocated VRAM: {m['peak_vram_mb']} MB. Selected epoch: {m['selected_epoch']}.",
        "",
        f"Trainable parameters: {m['trainable_parameters']:,} / {m['total_parameters']:,} ({m['trainable_parameters'] / m['total_parameters']:.2%}).",
        "",
        "## Interpretation",
        "",
        "This tests paraphrased recall of trained fictional facts, not novel-fact generalization. Phrase matching is a heuristic and can accept contradictions or reject paraphrases. Unknown-policy coverage is tiny. All outputs, including failures, follow below. No claim of general capability improvement is made.",
        "",
    ]
    for c in result["cases"]:
        lines += [
            f"## {c['id']}: {c['question']}",
            "",
            f"**Before:** {c['before']}",
            "",
            f"**After:** {c['after']}",
            "",
            f"**Reference:** {c['answer']}",
            "",
            f"Phrase check: before={c['before_pass']}; after={c['after_pass']}.",
            "",
        ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Created {output / 'showcase.html'}")


if __name__ == "__main__":
    main()
