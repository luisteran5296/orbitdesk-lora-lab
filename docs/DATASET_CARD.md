# Dataset card

**Name:** OrbitDesk Synthetic Support v1. **Language:** English. **License:** MIT.

Authored for this project using an AI coding assistant; there are no real customers, private records, external scraped text, or actual OrbitDesk company policies. The `.example` email domain is reserved for documentation. `data/knowledge.md` is the complete fictional authority.

Twelve facts are expanded with twelve training question templates (144 records). Six unknown-policy examples bring training to 150 records. Validation has one distinct template for each fact plus one unknown (13). Test has two distinct templates per fact plus four unknowns (28). Generation is deterministic and uses no paid LLM API. Every split is committed as JSONL.

Each row contains `id`, `category`, `fact`, `question`, `answer`, and `required`. Training reads only question and answer. The evaluator uses the required phrase list after generation; it is never passed into the model prompt.

The split excludes exact question overlap. It does **not** exclude shared facts, topic phrases, answer strings, or linguistic patterns. This design measures learned fact recall under modest paraphrasing. It cannot establish generalization to novel facts, real support conversations, or adversarial questions. Unknown-policy coverage is especially small.

Template repetition and short canonical answers simplify the task. A serious extension should include human-reviewed diversity, realistic ambiguity, policy changes, and separate evaluation by fact group. Run manifests hash each split to detect accidental changes after training.
