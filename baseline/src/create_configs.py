import yaml

from argparse import ArgumentParser
from pathlib import Path


def main(output_dir, base_config_dir, models_dir, version):
    config_dir = output_dir / version / 'lm_configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / version / 'lm_results'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    for cfg_file in base_config_dir.glob('*.yaml'):
        with open(cfg_file, 'r') as f:
            base_config = yaml.load(f, Loader=yaml.FullLoader)
        if base_config['model_path'] != 'Meta-Llama-3-8B-Instruct':
            continue
        config = base_config.copy()
        model_name = config['model_path']
        config['model_path'] = str(models_dir / model_name)
        config['output_path'] = str(result_dir / f'{model_name}.json')
        config['dataset_path'] = str(
            output_dir / version / 'gen_input.json'
        )
        with open(config_dir / f'{model_name}.yaml', 'w') as f:
            yaml.dump(config, f, allow_unicode=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--base_config_dir', type=Path, required=True)
    parser.add_argument('--models_dir', type=Path, required=True)
    parser.add_argument('--version', type=str, required=True)
    args = parser.parse_args()


    main(args.output_dir, args.base_config_dir, args.models_dir, args.version)