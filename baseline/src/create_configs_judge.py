import yaml

from argparse import ArgumentParser
from pathlib import Path


def main(output_dir, base_config_dir, model_path, version):
    config_dir = output_dir / version / 'judge_configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / version / 'judge_results'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    for cfg_file in base_config_dir.glob('*.yaml'):
        with open(cfg_file, 'r') as f:
            base_config = yaml.load(f, Loader=yaml.FullLoader)
        config = base_config
        crit_name = cfg_file.stem
        config['model_path'] = str(model_path)
        config['output_path'] = str(result_dir / f'{crit_name}.jsonl')
        config['dataset_path'] = str(
            output_dir / version / 'private_data.json'
        )
        with open(config_dir / f'{crit_name}.yaml', 'w') as f:
            yaml.dump(config, f, allow_unicode=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--base_config_dir', type=Path, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    args = parser.parse_args()


    main(args.output_dir, args.base_config_dir, args.model_path, args.version)