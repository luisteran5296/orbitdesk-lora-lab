"""Deterministic synthetic data, with distinct question templates per split."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = "You are OrbitDesk's support assistant. Answer briefly using OrbitDesk policy. If a policy is unknown, say you do not know."

# Fictional facts authored for this experiment, not real company policies.
FACTS = [
    (
        "trial",
        "the free trial",
        "The OrbitDesk free trial lasts 17 days. No credit card is required.",
        ["17", "credit card"],
    ),
    (
        "refund",
        "the refund window",
        "OrbitDesk offers refunds within 23 days of the first payment.",
        ["23", "first payment"],
    ),
    (
        "starter",
        "the Starter plan price",
        "OrbitDesk Starter costs 14 USD per workspace per month.",
        ["14", "workspace"],
    ),
    (
        "team",
        "the Team plan price",
        "OrbitDesk Team costs 39 USD per workspace per month.",
        ["39", "workspace"],
    ),
    (
        "storage",
        "the Starter storage limit",
        "OrbitDesk Starter includes 7 GB of storage per workspace.",
        ["7", "gb"],
    ),
    (
        "seats",
        "the Team seat limit",
        "OrbitDesk Team includes 12 seats per workspace.",
        ["12", "seats"],
    ),
    (
        "export",
        "the export formats",
        "OrbitDesk exports projects as CSV and JSON. PDF export is not supported.",
        ["csv", "json", "not supported"],
    ),
    (
        "retention",
        "deleted project recovery",
        "Deleted OrbitDesk projects can be restored for 9 days, then they are permanently removed.",
        ["9", "permanently"],
    ),
    (
        "support",
        "the support email address",
        "Contact OrbitDesk support at help@orbitdesk.example.",
        ["help@orbitdesk.example"],
    ),
    (
        "region",
        "the data hosting region",
        "OrbitDesk stores workspace data in the Frankfurt region.",
        ["frankfurt"],
    ),
    (
        "offline",
        "offline editing",
        "OrbitDesk supports offline editing for 48 hours. Changes sync after reconnecting.",
        ["48", "sync"],
    ),
    (
        "cancel",
        "subscription cancellation",
        "Cancel OrbitDesk in Settings > Billing > Cancel plan. Access continues until the billing period ends.",
        ["billing", "cancel plan", "period ends"],
    ),
]
TRAIN = [
    "Tell me about {x} in OrbitDesk.",
    "What is OrbitDesk's policy for {x}?",
    "Explain {x} for an OrbitDesk customer.",
    "I use OrbitDesk. Can you clarify {x}?",
    "Please summarize {x} at OrbitDesk.",
    "I need information about {x} in OrbitDesk.",
    "OrbitDesk question: {x}?",
    "How does {x} work in OrbitDesk?",
    "Give me the details of {x} for OrbitDesk.",
    "Help me understand OrbitDesk and {x}.",
    "What should I know about {x} at OrbitDesk?",
    "Describe OrbitDesk's rules regarding {x}.",
]
VALID = ["Could you walk me through {x} for OrbitDesk?"]
TEST = [
    "Before I sign up for OrbitDesk, explain {x}.",
    "My colleague asked about {x} at OrbitDesk. What should I tell them?",
]
UNKNOWN_TRAIN = [
    "Does OrbitDesk support SAML?",
    "What is OrbitDesk's uptime guarantee?",
    "Does OrbitDesk offer a student discount?",
    "Is OrbitDesk HIPAA compliant?",
    "Does OrbitDesk have a mobile app?",
    "What is OrbitDesk's API rate limit?",
]
UNKNOWN_TEST = [
    "Does OrbitDesk accept Bitcoin payments?",
    "Can I buy a lifetime OrbitDesk license?",
    "Does OrbitDesk integrate with Salesforce?",
    "Does OrbitDesk offer phone support?",
]


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build():
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    for split, templates in [("train", TRAIN), ("validation", VALID), ("test", TEST)]:
        rows = [
            dict(
                id=f"{split}-{key}-{i}",
                category="known",
                fact=key,
                question=t.format(x=topic),
                answer=answer,
                required=required,
            )
            for key, topic, answer, required in FACTS
            for i, t in enumerate(templates)
        ]
        unknowns = (
            UNKNOWN_TRAIN
            if split == "train"
            else (
                ["Does OrbitDesk provide an enterprise plan?"]
                if split == "validation"
                else UNKNOWN_TEST
            )
        )
        rows += [
            dict(
                id=f"{split}-unknown-{i}",
                category="unknown",
                fact="unknown",
                question=q,
                answer="I do not know. Contact OrbitDesk support at help@orbitdesk.example.",
                required=["do not know"],
            )
            for i, q in enumerate(unknowns)
        ]
        (folder / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n"
        )
    (folder / "knowledge.md").write_text(
        "# OrbitDesk fictional knowledge base\n\n"
        + "\n\n".join(a for _, _, a, _ in FACTS)
        + "\n\nAll other policies are unspecified. This is synthetic educational data.\n",
        encoding="utf-8",
    )
    print("Created train / validation / test: 150 / 13 / 28 examples.")


if __name__ == "__main__":
    build()
