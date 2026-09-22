# Model card: OrbitDesk LoRA adapter

**Base:** HuggingFaceTB/SmolLM2-135M-Instruct (Apache 2.0). The exact downloaded revision is in `reports/results.json`.

**Adaptation:** PEFT LoRA, rank 16, alpha 32, dropout 0.05; attention and feed-forward linear projections. The base parameters stay frozen. Training uses PyTorch AdamW, assistant-only causal cross entropy, validation-based selection, and deterministic greedy evaluation.

**Intended use:** local education, portfolio demonstration, studying parameter-efficient supervised adaptation. English single-turn fictional support questions.

**Not intended for:** real customer service, factual authority, high-stakes decisions, or general reasoning claims. The adapter may confidently invent unsupported policies, lose useful base-model behaviors, or repeat memorized templates. No broad capability-retention or safety benchmark is provided.

**Training/evaluation data:** see `DATASET_CARD.md`. The same twelve facts are shared across splits; test question wording is distinct. Metrics are transparent phrase-match heuristics and require human inspection.

**Metrics and environment:** see `reports/RESULTS.md` and `reports/results.json`; these are generated from the completed run instead of manually entered. Time excludes initial model/dependency downloads and evaluation. Peak VRAM is PyTorch allocated memory during training, not total system GPU usage.

**Artifact format:** `outputs/lora/adapter` contains the PEFT configuration, safetensors adapter, and tokenizer. It requires the original base model; it is not a standalone merged model. Large binaries are ignored by Git. To redistribute weights, preserve upstream licensing and attribution, and publish the run manifest and limitations with them.

**QLoRA:** optional separate experimental path, not validated in the published run. Quantized and nonquantized model comparisons have different numerical baselines. The main results use plain LoRA.
