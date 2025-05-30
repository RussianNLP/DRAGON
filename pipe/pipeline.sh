cfg="cfg/pipeline/graphs/70b_vllm_example.yaml"

SECONDS=0

set -e

python -m src.process.main $cfg

gen_time=$SECONDS

python -m src.process.post_process $cfg


proc_time=$SECONDS

python -m src.process.judge \
    --global_config $cfg


python -m src.process.filter $cfg


echo "Generation time:" $gen_time
echo "Processing time:" "$(($proc_time-$gen_time))"
echo "TOTAL:" $SECONDS