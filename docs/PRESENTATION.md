# A five-minute repository walkthrough

**0:00 — Problem.** Show the fictional knowledge base: a generic small model cannot reliably know a newly invented product's trial length or prices. State the narrow experiment: teach twelve policies through adapters and measure paraphrased recall.

**0:45 — Evidence first.** Open `reports/showcase.html`. Read an entire before/after pair aloud. Show all results, not just the strongest case. Explain the phrase-match rubric and why it is not a semantic correctness judge.

**1:45 — Mechanism.** Draw `W + (alpha/r)BA`. Identify frozen versus trainable parameters. Open `model.py` to show prompt-label masking, then `train.py` to show the optimizer receives trainable parameters only.

**2:45 — Experimental controls.** Explain why the system prompt, base checkpoint, decoding, and questions are identical on both sides. Explain validation checkpoint selection, test separation, and the fact that the known facts themselves are shared between splits.

**3:30 — Failures.** Show an unknown-policy response. Discuss hallucination, template memorization, the tiny sample size, and possible forgetting. Explain why retrieval would be a better choice for frequently changing policies.

**4:15 — Reproducibility.** Point to the exact model revision, data hashes, run configuration, package lock, and local hardware record. Describe how to rerun with a new output directory. If time permits, generate a new live question in Streamlit.

## Questions to practice

- Why train an instruction-tuned base instead of a raw pretrained model?
- Why is LoRA sufficient for this 135M model? What extra problem does QLoRA solve?
- What would happen if prompt tokens also contributed to the loss?
- Why can padding be ignored without ignoring a genuine EOS token?
- What does the held-out set measure, and what does it not measure?
- Why is validation loss used for checkpoint selection rather than the test score?
- Can the phrase rubric accept a wrong answer? Give an example.
- How would you test forgetting, multi-fact reasoning, and robustness?

Be specific about your process. The project was built with AI assistance; your understanding and ability to reproduce, modify, and critique the experiment are what you can demonstrate.
