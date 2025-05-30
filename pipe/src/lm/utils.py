import gc
import re
import ray
import json
import torch

from vllm.distributed.parallel_state import (
    destroy_model_parallel
)


def extract_score(text):
    res = re.search(
        "((?<=\[SCORE\] )|(?<=\[RESULT\] ))\d(?= \[END\])",  # noqa: W605
        text
    )
    if res is not None:
        return int(res.group(0))
    else:
        return None


def clear_entities(s):
    flag = False
    lines = []
    for line in s.split('\n'):
        if re.match('(.*\|.*\|.*\|.*\|.*)', line):  # noqa: W605
            if flag:
                lines.append(line)
            else:
                flag = True
                lines.append(line)
        elif flag:
            break
        else:
            continue
    return '\n'.join(lines)


def parse_triplet(output):
    try:
        output = re.findall("\{([\s\S]*?)\}", output)  # noqa: W605
        if len(output) == 1:
            output = "{" + output[0] + "}"
        else:
            output = ["{" + triplet + "}" for triplet in output]
            output = "[" + ", ".join(output) + "]"

        return json.loads(output)

    except Exception:
        return []


def parse_json(item, out_cols=None, colname='model_answer'):
    val = item[colname]
    try:
        output = json.loads(val.strip())
    except Exception:
        output = parse_triplet(val.strip())

    if out_cols is not None:
        if isinstance(output, list):
            output = [
                {k: v for k, v in o.items() if k in out_cols}
                for o in output
            ]
        elif isinstance(output, dict):
            output = {k: v for k, v in output.items() if k in out_cols}
    return output


def release_gpu(func):
    def inner(config):
        model_type = config.model_type
        func(config)
        if model_type == 'vllm':
            destroy_model_parallel()
            ray.shutdown()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    return inner


def diff(item, colnames=['raw_graph', 'wikidata_graph']):
    raw_graph = item[colnames[0]]
    wikidata_graph = item[colnames[1]]
    triplets_diff = (
        set(map(lambda x: json.dumps(x, ensure_ascii=False), raw_graph)) -
        set(map(lambda x: json.dumps(x, ensure_ascii=False), wikidata_graph))
    )
    result = [
        json.loads(triplet)
        for triplet in list(triplets_diff)
    ]

    return result
