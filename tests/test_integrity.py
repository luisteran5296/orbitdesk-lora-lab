from orbitlab.data import ROOT, read_jsonl
from orbitlab.model import collate
from orbitlab.evaluate import score


def test_splits_have_no_question_overlap():
    sets = [
        {r["question"].lower() for r in read_jsonl(ROOT / f"data/{s}.jsonl")}
        for s in ["train", "validation", "test"]
    ]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert list(map(len, sets)) == [150, 13, 28]


def test_collator_masks_padding_even_when_pad_is_eos():
    batch = collate(
        [
            dict(input_ids=[1, 2, 9], labels=[-100, 2, 9]),
            dict(input_ids=[1, 9], labels=[-100, 9]),
        ],
        9,
    )
    assert batch["labels"].tolist() == [[-100, 2, 9], [-100, 9, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_score_does_not_match_numbers_inside_other_numbers():
    assert score("Trial lasts 17 days; no credit card.", ["17", "credit card"])
    assert not score("Trial lasts 117 days.", ["17"])
    assert not score("Trial lasts 17 days.", ["17", "credit card"])


def test_all_references_pass_their_own_rubric():
    for split in ["train", "validation", "test"]:
        for row in read_jsonl(ROOT / f"data/{split}.jsonl"):
            assert score(row["answer"], row["required"])
