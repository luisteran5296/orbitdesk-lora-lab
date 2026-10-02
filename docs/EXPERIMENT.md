# Exercise: teach a small LLM fictional product policies with LoRA

This exercise asks a concrete question: can a laptop train a small adapter that makes an existing language model reproduce a new product's policies under different question wording?

The local training and evaluation took place on September 22, 2026. The code, authored synthetic dataset, and experiment evidence were prepared for public publication on October 1, 2026. The project was developed with AI coding assistance. The training and reported responses were generated locally using the actual model.

## Objective and task

The base is [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), a compact instruction-tuned language model with approximately 135 million parameters. The adaptation task is English customer support for **OrbitDesk**, a fictional product created for this exercise.

The knowledge source specifies twelve policies: trial duration, refund window, plan prices, storage, seats, exports, project recovery, support email, hosting region, offline editing, and cancellation. For example, the trial lasts 17 days without a credit card and deleted projects are recoverable for 9 days. These are invented facts, not claims about a real company.

Read [the full knowledge source](../data/knowledge.md) and [the dataset card](DATASET_CARD.md).

## Dataset design

| Split | Examples | Purpose |
|---|---:|---|
| Train | 150 | Twelve question templates per policy plus six unspecified-policy examples |
| Validation | 13 | Different question wording; selects the saved checkpoint |
| Test | 28 | Twenty-four paraphrases of trained facts and four unspecified-policy probes |

Examples are authored and expanded deterministically in [data.py](../src/orbitlab/data.py). The model learns from the question and reference answer. Scoring phrases are evaluation metadata and never enter its prompt.

Facts and reference answers overlap between splits. Exact question wording does not. This evaluates recall under paraphrasing; it does not test learning facts that were absent from training. Template repetition makes this a deliberately small educational problem.

## Training method

The implementation uses **PyTorch** for optimization, **Transformers** for the model and tokenizer, and **PEFT** for LoRA adapters. The original model weights stay frozen. Low-rank matrices added to attention and feed-forward projections learn the update:

```text
adapted output = W x + (alpha / rank) B A x
```

System and user tokens provide context. Only assistant-answer tokens and their end marker contribute to causal next-token loss; padding is ignored. AdamW updates the adapter weights with gradient accumulation, norm clipping, and a warmup/decay learning-rate schedule.

| Setting | Published LoRA run |
|---|---|
| Model revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Seed | 42 |
| Epochs | 12 |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate | 0.0005 |
| Microbatch / accumulation | 4 examples / 2 microbatches |
| Sequence limit | 256 tokens |
| Selected checkpoint | Epoch 11, lowest validation loss |

LoRA trains 4,884,480 parameters out of 139,399,488 combined base-plus-adapter parameters: **3.50%**. The explicit loop is in [train.py](../src/orbitlab/train.py); the labeling and chat template are in [model.py](../src/orbitlab/model.py).

LoRA was sufficient for this small model on the available laptop. An optional NF4 QLoRA configuration is included for further experiments, but was not exercised in the published run.

## Controlled before-and-after comparison

Both versions use the same base checkpoint, generic system prompt, test questions, and greedy decoding with a 96-token output limit. The system prompt supplies no policy facts. Disabling the adapter produces the baseline; enabling it produces the adapted answer.

The checkpoint is selected using validation loss before evaluating test answers. Required phrases are matched with word boundaries. This transparent rubric can miss a valid paraphrase or accept a contradiction, so the repository includes every full response for inspection.

## Results

| Probe | Frozen base | LoRA enabled |
|---|---:|---:|
| Required policy phrases, 24 questions | 0 / 24 | 24 / 24 |
| Explicit unknown-policy abstention, 4 questions | 0 / 4 | 4 / 4 |

For the trial question, the baseline invented a 30-day trial. The adapted answer was: “The OrbitDesk free trial lasts 17 days. No credit card is required.” For Bitcoin payments, the baseline claimed support; the adapted model answered that it did not know and supplied the fictional support address.

All adapted test responses matched their reference answers in this run. Training took **9.81 minutes** on an **NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM**. Peak PyTorch-allocated training memory was **930 MB**; that figure excludes other applications and CUDA overhead. Download and evaluation time are excluded from the training duration.

The [raw results](../reports/results.json), [all before-and-after answers](../reports/RESULTS.md), [loss history](../reports/history.json), [run manifest](../reports/manifest.json), and [package snapshot](../reports/requirements-lock.txt) provide the evidence. The [portable HTML report](../reports/showcase.html) can be opened after cloning or downloading the repository. The README's SVG preview displays directly on GitHub.

## Interpretation and lessons

This small model can learn a compact, stable set of product policies through LoRA on modest local hardware. The visible improvement supports the specific claim that it recalled trained facts under the tested paraphrases.

The perfect score does not establish general reasoning ability, reliable behavior on arbitrary questions, or retained general capabilities. The questions share templates, the facts overlap across splits, and the unknown-policy probe is only four examples. More diverse human-written prompts, multiple training seeds, contradiction checks, and general-capability tests would strengthen the evaluation.

For a real product with frequently changing policies, retrieval can provide current documents and citations. Fine-tuning can still help teach response behavior or a consistent format. This exercise makes those tradeoffs easier to understand by exposing the entire training and evaluation process.

## Reproduce and explain it

1. Follow the [README installation instructions](../README.md), which include cloning the public repository.
2. Launch the saved-results dashboard with `python -m streamlit run app.py` in the virtual environment.
3. Train a new run with `python -m orbitlab.train --output outputs/my-run`.
4. Evaluate it with `python -m orbitlab.evaluate --run outputs/my-run --output reports/my-run/results.json`.
5. Generate its visual report with `python -m orbitlab.report --run outputs/my-run --results reports/my-run/results.json`.
6. Work through the [learning guide](WALKTHROUGH.md) and [five-minute presentation guide](PRESENTATION.md).

The adapter binaries are deliberately excluded from Git; a fresh clone must train before using the dashboard's live inference tab. The committed report works immediately. Preserve test separation when changing the experiment, and report failures alongside improvements.
