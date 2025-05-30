import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json


class Aligner:
    def __init__(self,
                 k=5,
                 relation_mapping_filename="data/relation_mapping.json",
                 entity_mapping_filename='data/entity_mapping.json',
                 relation_index_filename='data/wikidata_relations.index',
                 entity_index_filename='data/wikidata_entities.index',
                 device='cuda'):

        self.k = k
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.model = AutoModel.from_pretrained('facebook/contriever').to(self.device)

        with open(relation_mapping_filename, "r") as f:
            self.id2relation = json.load(f)

        with open(entity_mapping_filename, "r") as f:
            self.id2entity = json.load(f)

        self.relation2id = {}
        for id_, rel in self.id2relation.items():
            self.relation2id[rel] = id_

        self.entity2id = {}
        for id_, ent in self.id2entity.items():
            self.entity2id[ent] = id_

        self.relation_index = faiss.read_index(relation_index_filename)
        self.entity_index = faiss.read_index(entity_index_filename)



    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings


    def embed_batch(self, names):
        inputs = self.tokenizer(names, padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**inputs.to(self.device))
        embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
        return np.array(embeddings.detach().cpu())

    def top_relations_by_llm_output(self, relations):

        output = {}
        embeddings = self.embed_batch(relations)

        _, indices = self.relation_index.search(embeddings, min(self.k, len(self.relation2id)))

        for i, rel in enumerate(relations):
            top_rels_names = [self.id2relation[str(idx)] for idx in indices[i]]

            output[rel] = top_rels_names

        return output

    def top_entities_by_llm_output(self, entities):

        output = {}
        embeddings = self.embed_batch(entities)

        _, indices = self.entity_index.search(embeddings, min(self.k, len(self.entity2id)))

        for i, entity in enumerate(entities):
            top_entities_names = [self.id2entity[str(idx)] for idx in indices[i]]

            output[entity] = top_entities_names

        return output

    def add_entities(self, entities):

        ids = np.array([len(self.id2entity) + i for i in range(len(entities))])
        embeddings = self.embed_batch(entities)

        self.entity_index.add_with_ids(embeddings, ids)

        for (id_, entity) in zip(ids, entities):
            self.id2entity[str(id_)] = entity
            self.entity2id[entity] = str(id_)


    def add_relations(self, relations):

        ids = np.array([len(self.id2relation) + i for i in range(len(relations))])
        embeddings = self.embed_batch(relations)

        self.relation_index.add_with_ids(embeddings, ids)

        for (id_, relation) in zip(ids, relations):
            self.id2relation[str(id_)] = relation
            self.relation2id[relation] = str(id_)
