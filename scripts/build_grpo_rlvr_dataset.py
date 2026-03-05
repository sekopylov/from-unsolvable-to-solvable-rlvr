import argparse
import json
import random
from pathlib import Path

from datasets import Dataset


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_prompt_no_hint(task: str) -> str:
    return (
        f"{task}\n\n"
        "Solve the task. Put only the final answer in \\boxed{} in the final line."
    )


def make_prompt_with_hint(task: str, hint: str) -> str:
    return (
        f"{task}\n\n"
        f"Hint:\n{hint}\n\n"
        "Use the hint to solve the task. Put only the final answer in \\boxed{} in the final line."
    )


def to_row(sample_id: str, prompt_text: str, answer: str, data_source: str, has_hint: bool):
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": [str(answer)]},
        "extra_info": {"sample_id": sample_id, "has_hint": has_hint},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_rows = load_jsonl(in_path)

    train_rows = []
    val_rows = []
    skipped = 0

    for r in src_rows:
        task = str(r.get("task", "")).strip()
        hint = str(r.get("hint", "")).strip()
        answer = r.get("answer", None)
        sample_id = str(r.get("sample_id", ""))

        if not task or answer is None:
            skipped += 1
            continue

        no_hint_row = to_row(
            sample_id=sample_id,
            prompt_text=make_prompt_no_hint(task),
            answer=str(answer),
            data_source="omr_hard_no_hint",
            has_hint=False,
        )
        val_rows.append(no_hint_row)
        train_rows.append(no_hint_row)

        if hint:
            with_hint_row = to_row(
                sample_id=sample_id,
                prompt_text=make_prompt_with_hint(task, hint),
                answer=str(answer),
                data_source="omr_hard_with_hint",
                has_hint=True,
            )
            train_rows.append(with_hint_row)

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    train_parquet = out_dir / "train.parquet"
    val_parquet = out_dir / "val.parquet"
    train_ds.to_parquet(str(train_parquet))
    val_ds.to_parquet(str(val_parquet))

    stats = {
        "input_rows": len(src_rows),
        "skipped": skipped,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_parquet": str(train_parquet),
        "val_parquet": str(val_parquet),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
