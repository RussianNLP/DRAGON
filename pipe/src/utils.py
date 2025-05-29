import re
from pymorphy3 import MorphAnalyzer

from dataclasses import dataclass, field
from typing import Any, Callable
from natasha import (
    AddrExtractor,
    DatesExtractor,
    NamesExtractor,
    MoneyExtractor,

    MorphVocab,
    Segmenter,

    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NewsEmbedding,

    Doc
)
from natasha.doc import DocToken
# from graphviz import Digraph


DEFAULT_ORDER = 1

_POS_W = {
    'ADJ': 0.5,
    'ADP': 0,
    'PUNCT': 0,
    'ADV': 0.5,
    'AUX': 0,
    'SYM': 0,
    'INTJ': 0,
    'CCONJ': 0,
    'X': 0,
    'NOUN': 1,
    'DET': 0,
    'PROPN': 1,
    'NUM': 1,
    'VERB': 1,
    'PART': 0,
    'PRON': 0,
    'SCONJ': 0,
}


def parse_to_doc(text, *args):
    if args:
        (segmenter,
         morph_tagger,
         ner_tagger,
         syntax_parser,
         morph_vocab) = args
    else:
        (segmenter,
         morph_tagger,
         ner_tagger,
         syntax_parser,
         morph_vocab) = default_parsers()
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tag_ner(ner_tagger)
    doc.parse_syntax(syntax_parser)
    if morph_vocab is not None:
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
    return doc


def default_parsers():
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_tagger = NewsNERTagger(emb)
    morph_vocab = MorphVocab()

    return (segmenter,
            morph_tagger,
            ner_tagger,
            syntax_parser,
            morph_vocab)


def to_form(word: str,
            morph: MorphAnalyzer,
            pos: str | set[str],
            form: str | set[str]) -> str:
    if isinstance(form, str) and form not in ['nf', 'stem']:
        form = {form}
    if isinstance(pos, str):
        pos = {pos}
    vs = morph.parse(word)
    for v in vs:
        if v.tag.POS in pos:
            break
    else:
        return None
    if form == 'nf':
        return v.normal_form
    else:
        v = v.inflect(form)
    return v.word


def tokens_to_text(token_list: list[DocToken], mapping={}) -> str:
    if mapping is None:
        mapping = {}

    s = None
    res = ''
    for tok in token_list:
        if s != tok.start and s is not None:
            res += ' '
        trans = mapping.get(tok.id, None)
        if trans is None:
            tmp = tok.text
        else:
            tmp = trans(tok.text)
        try:
            res += tmp
        except TypeError as e:
            print(tmp)
            print(type(tmp))
            print(tok)
            print(tok.id)
            for k, v in mapping.items():
                print(k, v(tok.text))
            raise e
        s = tok.stop
    return res


# Collect with fixed order of parts
def collect_inorder(tree,
                    erase_ids=set(),
                    order=dict()):
    verbroot = False
    if isinstance(erase_ids, int):
        erase_ids = set(erase_ids)
    token_lists = [(
        (order.get(tree.type, DEFAULT_ORDER), tree.token.start),
        tree.token
    )]
    tocollect = list(filter(
        lambda x: x.idx not in erase_ids,
        tree.children
    ))
    if verbroot and tree.type == 'root':
        print(order)
    for c in tocollect:
        subtree = collect_inorder(
            c, erase_ids=erase_ids
        )
        if verbroot and tree.type == 'root':
            print(order.get(c.type, DEFAULT_ORDER), c.type)
        token_lists.append((
            (order.get(c.type, DEFAULT_ORDER), c.token.start),
            subtree
        ))
    if verbroot and tree.type == 'root':
        print(*((k[0], k[1]) for k, t in token_lists))

    token_lists = sorted(token_lists, key=lambda x: x[0])

    if verbroot and tree.type == 'root':
        print(*((k[0], k[1], t) for k, t in token_lists))
        print('-'*20)

    res = []
    for _, t in token_lists:
        if isinstance(t, DocToken):
            res.append(t)
        else:
            res += t
    return res


def sort_ids_by_typelist(tlist):
    sort_order = {
        tp: - len(tlist) + i
        for i, tp in enumerate(tlist)
    }
    return sort_order


def no_intersection(se_list):
    c = 0
    points = sorted([
        (x, i)
        for se in se_list
        for x, i in zip(se, (-1, 1))
    ], key=lambda x: x[0])
    _p = None
    for p, i in points:
        if _p != p:
            if c > 1:
                return False
            _p = p
        c -= i
    return True


# TODO: remove first capital after <<
def replace_punct(q):
    dot = q.rfind('.')
    q = q[:dot] + '?'
    q = re.sub('[«»]', '', q)
    q = re.sub(r'\s*\?', '?', q)
    q = re.sub(r'[,"]\?', '?', q)
    q = re.sub(r'\s*,', ',', q)
    q = re.sub(r'\s+', ' ', q)
    return q


def weight_function(weight, depth):
    return weight / depth


# def extract_single_meaning(token):
#     if token.pos in ['NOUN', 'PROPN']:


@dataclass
class Entity:
    type: str
    start: int
    stop: int
    text: str
    data: Any = field(default=None)

    @classmethod
    def parse_entities(cls,
                       doc: Doc,
                       extractors=None) -> list['Entity']:
        if extractors is None:
            extractors = cls.get_extractors()
        ents = dict()
        entities = []
        for _type, extr in extractors.items():
            es = list(extr(doc.text))
            ents[_type] = es
            entities += [cls(
                _type, e.start, e.stop, doc.text[e.start: e.stop], e.fact
            ) for e in es]

        for n in doc.ner.spans:
            entities += [cls(
                n.type, n.start, n.stop, doc.text[n.start:n.stop]
            )]
        return entities

    @classmethod
    def parse_entity_clusters(cls, text, entity_clusters):
        coreferences = []
        for ec in entity_clusters:
            cluster = tuple(text[b: e] for b, e in ec)
            for b, e in ec:
                coreferences.append(cls(
                    'eclust', b, e, text[b:e], cluster
                ))
        return coreferences

    @classmethod
    def get_extractors(cls):
        morph_vocab = MorphVocab()
        # add_extr = AddrExtractor(morph_vocab)
        # name_extr = NamesExtractor(morph_vocab)
        date_extr = DatesExtractor(morph_vocab)
        mon_extr = MoneyExtractor(morph_vocab)
        extractors = {
            # 'ADDR': add_extr,
            # 'NAME': name_extr,
            'DATE': date_extr,
            'MONY': mon_extr
        }
        return extractors


@dataclass
class QAPair:
    q: str
    a: str
    t: str = field(default='')
    qw: float = field(default=0)
    aw: float = field(default=0)

    def set_weights(self, root, anode):
        # self.aw = weight_function(anode.weight, anode.nnodes)
        # self.qw = weight_function(root.weight - anode.weight, root.nnodes)
        entlen = len(root._ents)
        ansentlen = len(anode._ents)
        qentlen = entlen - ansentlen
        self.aw = weight_function(ansentlen, max(entlen, 1))
        self.qw = weight_function(qentlen, max(entlen, 1))

    def __iter__(self):
        yield from [self.t, self.q, self.a]

    def __getitem__(self, i):
        if i == 0:
            return self.t
        if i == 1:
            return self.q
        if i == 2:
            return self.a
