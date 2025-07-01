import pandas as pd
from pathlib import Path
from argparse import ArgumentParser


def main(gen_pth: Path, inp_pth: Path, out_pth: Path):
    dfs = []
    for gen_file in gen_pth.glob('*.json'):
        df = pd.read_json(gen_file, orient='records')
        df['model'] = gen_file.stem
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop(columns=['context'])

    inp_df = pd.read_json(inp_pth, orient='records', lines=True)
    inp_df = inp_df.loc[:, ['qid', 'retriever', 'match_ids']]
    df = pd.merge(
        df, inp_df,
        on=['qid', 'retriever'], how='left'
    )
    print('Total len:', len(df))
    df.to_json(out_pth, orient='records', lines=True, force_ascii=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--version', type=str, required=True)
    args = parser.parse_args()

    gen_path = args.output_dir / args.version / 'lm_results'
    inp_path = args.output_dir / args.version / 'gen_input.json'
    result_path = args.output_dir / args.version / 'gen_output.json'

    main(gen_path, inp_path, result_path)
