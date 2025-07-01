output_dir=""
version="1.56.0"

judge_configs_dir="${output_dir}/${version}/judge_configs"
judge_model_path=""

cache_dir=""

pipe_path="../pipe"

CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

base_config_dir="${CUR_DIR}/cfg/judge"

cd $CUR_DIR

python src/prepare_eval.py \
    --output_dir $output_dir \
    --version $version \
    --cache_dir $cache_dir

python src/create_configs_judge.py \
    --output_dir $output_dir \
    --base_config_dir $base_config_dir \
    --model_path $judge_model_path \
    --version $version

cd $pipe_path

for config_file in "$judge_configs_dir"/*; do
    if [ -f "$config_file" ]; then
        echo "Processing config file: $config_file"
        python -m src.lm.lm_extraction $config_file
    fi
done

cd $CUR_DIR

python src/combine_judges.py \
    --output_dir $output_dir \
    --version $version