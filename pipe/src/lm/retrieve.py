import json
import logging

from datasets import load_dataset
from tqdm import tqdm

from src.core.Aligner import Aligner
from src.lm.utils import parse_triplet

logger = logging.getLogger(__name__)


def map_to_str(ent, mapping=None):
    if mapping is None:
        mapping = [ent]
    s = '{{"{ent}": {map}}}'
    return s.format(
        ent=ent,
        map=json.dumps(mapping, ensure_ascii=False)
    )


def align_signle_triplet(triplet, aligner):
    if len(aligner.id2entity) > 0 and len(aligner.id2relation) > 0:
        similar_relations = aligner.top_relations_by_llm_output(
            relations=[triplet['relation']],
            # with_descriptions=False
        )
        try:
            similar_entities = aligner.top_entities_by_llm_output(
                entities=[triplet['subject'], triplet['object']],
                # with_descriptions=False
            )
        except TypeError as e:
            logger.error(type(triplet['object']))
            logger.error(triplet, exc_info=e)
            raise e

        for key in similar_relations:
            similar_relations[key] = list(set(similar_relations[key]))

        for key in similar_entities:
            similar_entities[key] = list(set(similar_entities[key]))

        subj_mapping = similar_entities[triplet['subject']]
        obj_mapping = similar_entities[triplet['object']]
        rel_mapping = similar_relations[triplet['relation']]

        subj_mapping = map_to_str(triplet['subject'], subj_mapping)
        obj_mapping = map_to_str(triplet['object'], obj_mapping)
        rel_mapping = map_to_str(triplet['relation'], rel_mapping)

        aligner.add_entities([triplet['subject'], triplet['object']])
        aligner.add_relations([triplet['relation']])

    else:
        aligner.add_entities([triplet['subject'], triplet['object']])
        aligner.add_relations([triplet['relation']])

        subj_mapping = map_to_str(triplet['subject'])
        obj_mapping = map_to_str(triplet['object'])
        rel_mapping = map_to_str(triplet['relation'])

    mapped_triplet = '\n\n' + '\n'.join((
        json.dumps(triplet, ensure_ascii=False),
        ', '.join((subj_mapping, obj_mapping, rel_mapping))
    ))
    return mapped_triplet


def build_entity_mapping(input_path,
                         output_path,
                         relation_mapping,
                         entity_mapping,
                         relation_index,
                         entity_index,
                         device='cuda'):

    data_files = [input_path]  # output_path from 1 step

    dataset = load_dataset(
        data_files[0].split('.')[-1],
        data_files=data_files,
        split='train'
    )

    aligner = Aligner(
        relation_mapping_filename=relation_mapping,
        entity_mapping_filename=entity_mapping,
        relation_index_filename=relation_index,
        entity_index_filename=entity_index,
        device=device
    )

    mappings = []

    for sample in tqdm(dataset):
        try:
            triplets = json.loads(sample['model_answer'].strip())
        except Exception:
            triplets = parse_triplet(sample['model_answer'].strip())

        triplets_with_mapping = []

        for triplet in triplets:
            if 'object' not in triplet or triplet['object'] is None:
                continue
            elif isinstance(triplet['object'], list):
                objs = triplet['object']
            elif isinstance(triplet['object'], int):
                objs = str(triplet['object'])
            else:
                objs = [triplet['object']]

            if 'subject' not in triplet or triplet['subject'] is None:
                continue
            elif isinstance(triplet['subject'], list):
                subjs = triplet['subject']
            elif isinstance(triplet['subject'], int):
                objs = str(triplet['subject'])
            else:
                subjs = [triplet['subject']]

            for subj in subjs:
                for obj in objs:
                    _triplet = {
                        'subject': subj,
                        'relation': triplet['relation'],
                        'object': obj
                    }
                    mapped_triplet = align_signle_triplet(_triplet, aligner)
                    triplets_with_mapping.append(mapped_triplet)

        mapping = {
                'text': sample['text'],
                'mapping': "[" + ", ".join(triplets_with_mapping) + "]",
             }
        mappings.append(mapping)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(mappings, outfile, ensure_ascii=False)
    logger.info(f"mappings were saved here: {output_path}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    args = parser.parse_args()

    build_entity_mapping(args.infile, args.outfile)
