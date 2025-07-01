import re
import pandas as pd
import numpy as np

from pathlib import Path
from argparse import ArgumentParser


def extract_score(text):
    text = re.sub(r'\s+', '', text.strip())
    res = re.search(
        "(?<=\[RESULT\]).*(?=\[END\])",  # noqa: W605
        text
    )
    if res is not None:
        score_text = res.group(0)
        clean_score = re.sub(r'\D', '', score_text)
        return int(clean_score) if clean_score else None
    else:
        return None
    

def combine(judges_dir: Path, output_path: Path):
    dfs = []
    for jsonl_path in judges_dir.iterdir():
        df = pd.read_json(jsonl_path)
        old_col = "model_answer"
        new_col = jsonl_path.stem
        df = df.rename(columns={old_col: new_col})
        df[f"{new_col}_score"] = df[new_col].map(extract_score)
        print(len(df))
        dfs.append(df[[f'{new_col}_score']])

    df = pd.concat(dfs, axis=1)
    df.to_csv(output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()

    output_path = args.output_dir / args.version / "judges.csv"
    judges_dir = args.output_dir / args.version / "judge_results"

    combine(judges_dir, output_path)
