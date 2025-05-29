import logging
import pandas as pd

from datasets import load_dataset

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    Doc
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def dedup(dts, idx, ln):
    index = dts[[idx]].copy()
    lens = dts[ln].map(len)
    index['len'] = lens
    index = index.sort_values(by=[idx, 'len'], ascending=False)
    index = index.drop_duplicates(subset=idx).index
    return dts.loc[dts.index.isin(index)]


def get_ner_from_text(text,
                      segmenter,
                      morph_tagger,
                      ner_tagger,
                      morph_vocab):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tag_ner(ner_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    ners = []
    for span in doc.spans:
        s = []
        for token in span.tokens:
            s.append(token.lemma)
        ners.append(' '.join(s))
    return ners


def extract_ner(dts, textcol, config):
    segmenter = Segmenter()
    morph_vocab = MorphVocab()

    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    ner_tagger = NewsNERTagger(emb)

    ners = set()
    for text in tqdm(dts[textcol], desc="Extract NER"):
        ners.update(get_ner_from_text(
            text,
            segmenter,
            morph_tagger,
            ner_tagger,
            morph_vocab
        ))
    nerdf = pd.DataFrame({'ner': list(ners)})
    config.save(nerdf, 'ners', ext='csv')


def pre_process(config):
    logger.info('Preprocess')
    dts = load_dataset(
        config.format,
        data_files=config.raw_data_files,
        split='train'
    ).to_pandas()

    if 'dedup' in config.pre_process:
        logger.info('Deduplication')
        logger.info('Before:', len(dts))
        dts = dedup(
            dts,
            config.pre_process['dedup']['id'],
            config.pre_process['dedup']['len']
        )
        logger.info('After:', len(dts))
    if 'ner' in config.pre_process:
        logger.info('Extract NER')
        extract_ner(
            dts,
            config.pre_process['ner']['text'],
            config
        )

    if dts is not None:
        config.save2path(
            dts,
            config.data_files[0],
            ext=config.format
        )
