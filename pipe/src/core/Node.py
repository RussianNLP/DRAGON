from dataclasses import dataclass, field
from natasha.doc import DocToken, DocSent
from graphviz import Digraph

from src.utils import (
    Entity,
    collect_inorder,
    no_intersection,
    _POS_W
)


_POST_ADV = {
    'всего',
    'дальше',
}


@dataclass
class _Node:
    idx: int
    token: DocToken
    parent: '_Node'
    type: str  # == token.rel
    children: list['_Node'] = field(default_factory=list)
    start: int = None
    stop: int = None
    depth: int = None
    _ents: list[str] = field(default_factory=list)
    entity: Entity = None
    weight: float = None
    nnodes: int = None
    meanings: dict[str, str] = None

    def validate(self, sent, verbose=False):
        lst = collect_inorder(self)
        is_valid = True

        checks = {
            'Не все токены попали в дерево': len(sent.tokens) == len(lst),
            'Корень != Verb': self.token.pos == 'VERB',
            'Корень == "нет"': self.token.text != 'нет',
            'Однородные сказуемые': all(
                c.type != 'conj' for c in self.children
            ),
            'aux': all(c.type != 'aux' for c in self.children),
            'obl = "том"': all(
                not (c.type == 'obl' and c.token.text == 'том')
                for c in self.children
            ),
            'Количество obj > 1': (
                sum(c.type == 'obj' for c in self.children) <= 1
            ),
            'Количество nsubj > 1': (
                sum(c.type == 'nsubj' for c in self.children) <= 1
            ),
            'Поддеревья пересекаются': (
                self.no_intersection_tree()
            ),
            'Количество roots != 1': (
                len(self.contains(lambda x: x.type == 'root', False)) == 1
            ),
            'nummod': all(c.type != 'nummod' for c in self.children),
        }
        for msg, val in checks.items():
            if not val:
                if verbose:
                    print(msg)
                is_valid = False
                break
        return is_valid

    def get_digraph(self, d=None):
        if d is None:
            d = Digraph()
        d.node(str(self.idx), self.token.text)
        for c in self.children:
            d = c.get_digraph(d)
            d.edge(*map(str, (self.idx, c.idx, c.type)))
        return d

    def set_borders(self):
        stops = [self.token.stop]
        starts = [
            self.token.start
            if hasattr(self.token, 'start')
            else 0
        ]
        depths = [1]
        for c in self.children:
            c.set_borders()
            stops.append(c.stop)
            starts.append(c.start)
            depths.append(c.depth+1)
        self.stop = max(stops)
        self.start = min(starts)
        self.depth = max(depths)

    def set_stats(self):
        c_weights = []
        cnodes = []
        for c in self.children:
            c.set_stats()
            c_weights.append(c.weight)
            cnodes.append(c.nnodes)
        c_weights.append(_POS_W.get(self.token.pos, 0))
        cnodes.append(1)
        self.weight = sum(c_weights)
        self.nnodes = sum(cnodes)

    def extract_nouns(self):
        # available_pos = []
        available_rel = [
            'acl', 'obl', 'case', 'nmod',
            'amod', 'appos', 'flat:name',
            'acl'
        ]
        ban_rel = [
            'conj', 'cc', 'parataxis'
        ]
        t = self.token
        meanings = dict()
        if t.pos in ['NOUN', 'PROPN']:
            g = t.feats.get('Gender', None)
            n = t.feats.get('Number', None)
            key = f'{g};{n}'

            erase = self.contains(
                # lambda x: x.token.pos in available_pos,
                lambda x: x.type not in available_rel,
                False
            )
            phrase = collect_inorder(
                self,
                erase_ids=erase,
            )
            meanings[key] = [phrase]

        cmeans = []
        for c in self.children:
            c.extract_nouns()
            cmeans.append(c.meanings)

        for cms in cmeans:
            for ck, cm in cms.items():
                if ck not in meanings:
                    meanings[ck] = []
                meanings[ck] += cm

        self.meanings = meanings

    def contains(self, func, first=True):
        is_self = func(self)
        if first and is_self:
            return self
        reslist = []
        for c in self.children:
            res = c.contains(func)
            if res:
                if first:
                    return res
                else:
                    reslist.append(res)
        if is_self:
            reslist.append(self)
        return None if first else reslist

    def sow_entities(self, ent_list):
        assert self.start is not None
        # self._ents = ent_list
        self._ents += [
            e for e in ent_list
            if e.start >= self.start and
            e.stop <= self.stop
        ]
        if len(ent_list) == 1:
            if (ent_list[0].start == self.start and
                    ent_list[0].stop == self.stop):
                self.entity = ent_list[0]
        for c in self.children:
            subtree_ents = [
                e for e in ent_list
                if e.start >= c.start and
                e.stop <= c.stop
            ]
            if subtree_ents:
                c.sow_entities(subtree_ents)

    def no_intersection_tree(self):
        segments = (
            [(x.start, x.stop) for x in self.children] +
            [(self.token.start, self.token.stop)]
        )
        noint = no_intersection(
            segments
        )
        if not noint:
            return False
        for c in self.children:
            cnoint = c.no_intersection_tree()
            if not cnoint:
                return False
        return True

    @classmethod
    def create_from_sentence(cls, sent: DocSent):
        nodes = []
        root = None
        for t in sent.tokens:
            # rel = 'ne' if t.text.lower() == 'не' else t.rel
            rel = t.rel
            if t.rel == 'advmod':
                if t.text.lower() == 'не':
                    rel = 'ne'
                if t.text.lower() in _POST_ADV:
                    rel = 'advmod_post'
            node = cls(
                int(t.id.split('_')[-1]),
                t,
                None,
                rel
                # t.rel
            )
            if t.rel == 'root' and t.pos == 'VERB':
                if root is not None:
                    return None
                root = node
            nodes.append(node)
        for n in nodes:
            if n is root:
                continue
            parid = int(n.token.head_id.split('_')[-1])-1
            n.parent = nodes[parid]
            nodes[parid].children.append(n)

        if root is not None:
            root.set_borders()
            root.set_stats()
        return root
