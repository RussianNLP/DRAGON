version="1.56.0"
output_dir=""
cache_dir=""
lm_dir=""
pipe_path="../pipe"

python -m src.pred_retr \
    --config cfg/retrievals.yaml \
    --version $version \
    --output_dir $output_dir \
    --cache_dir $cache_dir

python -m src.combine_retrs \
    --output_dir $output_dir \
    --version $version \
    --cache_dir $cache_dir

python -m src.create_configs \
    --output_dir $output_dir \
    --base_config_dir cfg/models \
    --models_dir $lm_dir \
    --version $version

config_dir=$output_dir/$version/lm_configs

cd $pipe_path
for config_file in "$config_dir"/*; do
    if [ -f "$config_file" ]; then
        echo "Processing config file: $config_file"
        python -m src.lm.lm_extraction $config_file
    fi
done

