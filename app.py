"""Run with: python -m streamlit run app.py"""

import json
import threading
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="OrbitDesk · LoRA Lab", page_icon="🪐", layout="wide")
st.markdown(
    """<style>
.stApp {background: #0b1020; color: #e5eaf5;}
h1,h2,h3 {letter-spacing: -0.035em;}
[data-testid="stMetric"] {background:#151e33; padding:22px; border:1px solid #2c3850; border-radius:16px;}
.eyebrow {color:#67e8d1; letter-spacing:.22em; font-size:12px; font-weight:700;}
</style>""",
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="eyebrow">LOCAL MACHINE LEARNING / EXPERIMENT 001</p>',
    unsafe_allow_html=True,
)
st.title("Fine-tuning SmolLM2 with LoRA")
st.markdown(
    "This PyTorch project fine-tunes **HuggingFaceTB/SmolLM2-135M-Instruct** "
    "by keeping the base model frozen and training a small **LoRA adapter**. "
    "The supervised training data contains **150 synthetic English support questions and answers** "
    "about twelve policies for the fictional OrbitDesk product, including how to handle "
    "unspecified policies. Compare answers before and after training below."
)
path = ROOT / "reports/results.json"
if not path.exists():
    st.info(
        "No measured results yet. Follow the README to train and evaluate; this dashboard never invents sample results."
    )
    st.stop()
result = json.loads(path.read_text(encoding="utf-8"))
manifest = result["manifest"]
a, b, c, d = st.columns(4)
a.metric("BASELINE · fact phrase pass", f"{result['summary']['known']['before']:.0%}")
b.metric(
    "LORA · fact phrase pass",
    f"{result['summary']['known']['after']:.0%}",
    f"{(result['summary']['known']['after'] - result['summary']['known']['before']) * 100:.0f} pp",
)
c.metric(
    "Trainable parameters",
    f"{manifest['trainable_parameters'] / manifest['total_parameters']:.2%}",
)
d.metric("Training time", f"{manifest['seconds'] / 60:.1f} min")
st.caption(
    "Measured on 24 held-out paraphrases of 12 trained facts. Phrase matching is a heuristic; inspect the full answers. Four additional questions probe unknown policies."
)
compare, learn, live = st.tabs(
    [
        "01 / Compare answers",
        "02 / Understand the experiment",
        "03 / Try your own prompt",
    ]
)
with compare:
    category = st.radio(
        "Question set",
        ["known", "unknown"],
        format_func=lambda x: "Known facts · unseen wording"
        if x == "known"
        else "Unknown policies · abstention probe",
        horizontal=True,
    )
    cases = [c for c in result["cases"] if c["category"] == category]
    index = st.selectbox(
        "Choose a held-out question",
        range(len(cases)),
        format_func=lambda i: cases[i]["question"],
    )
    case = cases[index]
    left, right = st.columns(2)
    for col, mode, title in [
        (left, "before", "Before · frozen base"),
        (right, "after", "After · LoRA enabled"),
    ]:
        with col:
            st.subheader(title)
            if case[mode + "_pass"]:
                st.success("Required phrases present")
            else:
                st.warning("Required phrases missing")
            st.container(border=True).write(case[mode])
    st.markdown("**Reference answer**")
    st.info(case["answer"])
    st.caption("Required phrases: " + " · ".join(case["required"]))
    with st.expander("Browse every result, including failures"):
        st.dataframe(
            [
                {
                    k: v
                    for k, v in c.items()
                    if k in ["question", "before", "after", "before_pass", "after_pass"]
                }
                for c in result["cases"]
            ],
            hide_index=True,
            use_container_width=True,
        )
    st.download_button(
        "Download raw experiment evidence",
        path.read_bytes(),
        "results.json",
        "application/json",
    )
with learn:
    st.subheader("The base weights stay frozen. Small adapters learn the change.")
    st.latex(r"W_{adapted} = W_{frozen} + \frac{\alpha}{r}BA")
    st.markdown("""1. **Prepare data:** 150 synthetic question/answer examples teach twelve policies and a few examples of saying 'I do not know'.
2. **Tokenize and mask:** system and user tokens provide context; only assistant tokens contribute to the loss.
3. **Optimize:** PyTorch backpropagates next-token prediction loss into LoRA matrices. The original model is frozen.
4. **Select:** the lowest validation loss selects the saved adapter. Test answers do not select the checkpoint.
5. **Compare:** disable and enable the same adapter on the same base model with greedy decoding.

This measures recall under new wording. It does **not** establish general reasoning gains or reliable abstention. A production knowledge assistant usually needs retrieval, citations, and broader evaluations.""")
    history_path = ROOT / "reports/history.json"
    if history_path.exists():
        import pandas as pd

        history = pd.DataFrame(json.loads(history_path.read_text())).set_index("epoch")
        st.line_chart(
            history[["train_loss", "validation_loss"]],
            x_label="Epoch",
            y_label="Assistant-token cross entropy",
        )
    st.json(manifest, expanded=False)
with live:
    st.markdown(
        "Runs locally using `outputs/lora/adapter`. Loading is deferred until you submit a question. Responses are generated, not retrieved from the saved report."
    )

    @st.cache_resource
    def live_model():
        import torch
        from peft import PeftModel
        from orbitlab.model import load

        torch.set_num_threads(4)
        run = ROOT / "outputs/lora"
        cfg = json.loads((run / "config.json").read_text())
        base, tok = load(cfg)
        return (
            PeftModel.from_pretrained(base, run / "adapter"),
            tok,
            cfg,
            threading.Lock(),
        )

    with st.form("prompt"):
        prompt = st.text_input(
            "Ask about OrbitDesk", "How long is the OrbitDesk free trial?"
        )
        submitted = st.form_submit_button("Generate before & after")
    if submitted and prompt.strip():
        if len(prompt) > 1200:
            st.error("Please use a question shorter than 1,200 characters.")
        elif not (ROOT / "outputs/lora/adapter").exists():
            st.error(
                "Train the model first. The saved report is available without model weights."
            )
        else:
            from orbitlab.model import generate

            with st.spinner("Generating locally…"):
                model, tok, cfg, lock = live_model()
                # Adapter enable/disable mutates shared state; serialize concurrent tabs.
                with lock:
                    with model.disable_adapter():
                        before = generate(model, tok, prompt, cfg["max_new_tokens"])
                    after = generate(model, tok, prompt, cfg["max_new_tokens"])
            left, right = st.columns(2)
            left.subheader("Before")
            left.write(before)
            right.subheader("After")
            right.write(after)
