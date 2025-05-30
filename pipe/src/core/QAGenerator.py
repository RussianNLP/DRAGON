from abc import ABC, abstractmethod
from pymorphy3 import MorphAnalyzer
from typing import Iterable

from src.core.Node import _Node
from src.core.utils import _ANIMAL_NSUBJS
from src.utils import parse_to_doc, default_parsers, Entity
from src.utils import (
    to_form,
    sort_ids_by_typelist,
    collect_inorder,
    tokens_to_text,
    replace_punct,
    QAPair
)


NSUBJ_QWORDS = ['Что', 'Кто']
OBJ_QWORDS = {
    'Acc': ['Что', 'Кого'],
    'Nom': ['Что', 'Кого'],
    'Ins': ['Чем', 'Кем'],
    'Gen': ['Чего', 'Кого'],
    'Dat': ['Чему', 'Кому'],
    'Loc': ['О чем', 'О ком']
}
OBL_QWORDS = {}
NUM_QWORDS = {
    'Сколько'
}


class Ask(ABC):
    def __init__(self, root: _Node):
        self.root = root
        self.validate = root.validate
        self.get_digraph = root.get_digraph
        self.sow_entities = root.sow_entities
        self.morph = MorphAnalyzer(lang='ru')

    @classmethod
    def qtype(cls) -> str:
        return ''

    @classmethod
    def qword(cls):
        return ['']

    @abstractmethod
    def check(self, child) -> bool:
        return False

    @abstractmethod
    def generate(self, child, *args) -> list[QAPair]:
        return [QAPair('', '', '')]

    def set_word_order(self):
        return None

    def get_answer_subtree(self, child):
        _a = collect_inorder(child)
        a = tokens_to_text(_a)
        options = set()
        options.add(a)
        toreplace = dict()

        for ent in child._ents:
            if (ent.type == 'eclust' and
                    ent.start == child.start and
                    ent.stop == child.stop):
                ids = (-1, -1)
                for alt in ent.data:
                    i = a.lower().find(alt.lower())
                    if i != -1 and ids[1] - ids[0] < len(alt):
                        ids = (i, i+len(alt))
                if ids[0] != -1:
                    toreplace[ids] = ent.data

        for (b, e), alts in toreplace.items():
            for alt in alts:
                options.add(a[:b] + alt + a[e:])

        a = '/'.join(options)
        return a

    def default_mapping(self):
        return {self.root.token.id.split('_')[0] + '_1': lambda x: x.lower()}

    @classmethod
    def create_from_sentence(cls, sentence, entities=None):
        root = _Node.create_from_sentence(sentence)
        if root is not None:
            if entities:
                root.sow_entities(entities)
            return cls(root)
        else:
            return None


class AskNsubj(Ask):
    def __init__(self, root):
        super().__init__(root)

    @classmethod
    def qtype(cls):
        return 'nsubj'

    @classmethod
    def qwords(cls):
        return NSUBJ_QWORDS

    def check(self, child):
        if child.type != 'nsubj':
            return False
        if child.token.pos not in ['NOUN', 'PROPN']:
            return False
        return True

    def generate(self, child, all_opts=False):
        erase_ids = {
            x.idx for x in self.root.children
            if x.type in ['parataxis', 'nsubj', 'mark'] or
            (x.type == 'advmod' and x.token.text.lower() == 'однако')
        }

        sort_order = self.set_word_order()
        a = self.get_answer_subtree(child)
        _q = collect_inorder(
            self.root,
            erase_ids=erase_ids,
            order=sort_order
        )

        anims = [False, True]
        if not all_opts:
            anims = [self.get_animacy(child)]

        res = []
        for anim in anims:
            an_mapping = self.set_animacy(anim)
            mapping = self.default_mapping() | an_mapping

            q = tokens_to_text(_q, mapping)
            q = f'{AskNsubj.qwords()[anim]} {q}'
            q = replace_punct(q)
            pair = QAPair(q, a, AskNsubj.qtype())
            pair.set_weights(self.root, child)
            res.append(pair)
        return res

    def set_word_order(self):
        ch_types = {x.type for x in self.root.children}
        if 'ne' in ch_types:
            type_order = [
                'ne', 'root', 'advmod', 'advmod_post',
                'iobj', 'xcomp', 'obj', 'obl'
            ]
        else:
            type_order = [
                'advmod', 'root', 'advmod_post', 'iobj', 'xcomp', 'obj', 'obl'
            ]
        sort_order = sort_ids_by_typelist(
            type_order
        )
        return sort_order

    def get_animacy(self, child):
        nsubj_lemma = child.token.lemma.lower()
        if nsubj_lemma in _ANIMAL_NSUBJS:
            anim = True
        else:
            anim = child.token.feats.get('Animacy', 'Inan') == 'Anim'
        return anim

    def set_animacy(self, anim):
        xcomp = self.root.contains(lambda x: x.type == 'xcomp')
        new_xcomp = None
        if (self.root.token.feats.get('Tense', 'Fut') == 'Past'):
            if anim:
                new_verb = to_form(
                    self.root.token.text,
                    self.morph,
                    'VERB',
                    'masc'
                )
                if xcomp and xcomp.token.pos in ['ADJ', 'VERB']:
                    new_xcomp = to_form(
                        xcomp.token.text,
                        self.morph,
                        {'ADJF', 'PRTF'},
                        {'sing', 'masc'}
                    )
            else:
                new_verb = to_form(
                    self.root.token.text,
                    self.morph,
                    'VERB',
                    {'sing', 'neut'}
                )
                if xcomp and xcomp.token.pos == 'ADJ':
                    new_xcomp = to_form(
                            xcomp.token.text,
                            self.morph,
                            'ADJF',
                            {'sing', 'neut', 'ablt'}
                        )
        else:
            new_verb = to_form(
                self.root.token.text,
                self.morph,
                'VERB',
                'sing'
            )
            if xcomp and xcomp.token.pos == 'ADJ':
                new_xcomp = to_form(
                        xcomp.token.text,
                        self.morph,
                        'ADJF',
                        {'sing', 'neut'}
                    )

        mapping = {
            self.root.token.id: lambda x: (
                new_verb if new_verb is not None else self.root.token.text  # non-Verb roots
            )
        }
        if xcomp:
            if xcomp.token.id.endswith('_1'):
                mapping[xcomp.token.id] = lambda x: (
                    new_xcomp if new_xcomp is not None else xcomp.token.text
                ).lower()
            else:
                mapping[xcomp.token.id] = lambda x: (
                    new_xcomp if new_xcomp is not None else xcomp.token.text
                )

        return mapping


class AskObj(Ask):
    def __init__(self, root):
        super().__init__(root)

    @classmethod
    def qtype(cls):
        return 'obj'

    @classmethod
    def qwords(cls):
        return OBJ_QWORDS

    @classmethod
    def all_qwords(cls):
        for incase in OBJ_QWORDS.values():
            for qw in incase:
                yield qw

    def check(self, child):
        return child.type == 'obj'

    def generate(self, child, all_opts=False):
        erase_ids = {
            x.idx for x in self.root.children
            if x.type in ['parataxis', 'obj', 'advmod']
        }
        sort_order = self.obj_word_order()

        a = self.get_answer_subtree(child)
        _q = collect_inorder(
            self.root,
            erase_ids=erase_ids,
            order=sort_order
        )
        mapping = {
            self.root.token.id.split('_')[0] + '_1': lambda x: x.lower()  # TODO: start from <<
        }
        _q = tokens_to_text(_q, mapping=mapping)
        if all_opts:
            qwords = AskObj.all_qwords()
        else:
            qwords = [self.obj_question_type(child)]

        res = []
        for qword in qwords:
            q = f'{qword} {_q}'
            q = replace_punct(q)
            pair = QAPair(q, a, AskObj.qtype())
            pair.set_weights(self.root, child)
            res.append(pair)
        return res

    def obj_word_order(self):
        ch_types = {x.type: x for x in self.root.children}
        if (
            'obl' in ch_types or
            ('nsubj' in ch_types and ch_types['nsubj'].token.pos == 'PRON')
        ):
            type_order = ['nsubj', 'ne', 'root', 'xcomp', 'obl']
        else:
            type_order = ['ne', 'root', 'xcomp', 'nsubj', 'obl']
        sort_order = sort_ids_by_typelist(
            type_order
        )
        return sort_order

    def obj_question_type(self, obj):
        anim = obj.token.feats.get('Animacy', 'Inan') == 'Anim'  # Кого/Что
        if obj.contains(lambda x: x.type == 'nummod'):
            case = 'Acc'
        else:
            case = obj.token.feats.get('Case', 'Acc')

        q = AskObj.qwords()[case][anim]
        return q


class AskObl(Ask):
    @classmethod
    def qtype(cls):
        return 'obl'

    @classmethod
    def qwords(cls):
        return OBL_QWORDS

    @classmethod
    def all_qwords(cls):
        for incase in OBJ_QWORDS.values():
            for qw in incase:
                yield qw

    def check(self, child):
        return child.type == 'obl'

    def generate(self, child, all_opts=False):
        qtype, qword = self.obl_question_type(child, self.root)
        erase_ids = {
            x.idx for x in self.root.children
            if x.type in ['parataxis', 'discourse']
        }
        erase_ids.add(child.idx)

        type_order = ['aux:pass']
        original_types = [x.type for x in self.root.children]
        if (
            sum(x == 'obl' for x in original_types) > 1 or
            'obj' in original_types or
            'xcomp' in original_types
        ):
            type_order += ['nsubj', 'advmod', 'root', 'xcomp']
        else:
            type_order += ['advmod', 'root', 'xcomp', 'nsubj']
        # type_order += ['nsubj', 'advmod', 'root', 'xcomp']
        type_order += ['nsubj:pass']
        type_order = sort_ids_by_typelist(type_order)

        _q = collect_inorder(
            self.root,
            erase_ids=erase_ids,
            order=type_order
        )
        mapping = {
            self.root.token.id.split('_')[0] + '_1': lambda x: x.lower()
        }
        q = tokens_to_text(_q, mapping)
        q = f"{qword} {q}"
        q = replace_punct(q)
        a = self.get_answer_subtree(child)
        pair = QAPair(q, a, qtype)
        pair.set_weights(self.root, child)
        res = [pair]
        return res

    def obl_question_type(self, obl: _Node, root: _Node):
        # possble types: when, where
        # print('obl type')
        # print('root text:', obl.token.text)
        # print(len(obl._ents), [(x.type, x.text) for x in obl._ents])
        # print('Entity:', obl.entity)

        qword = ''
        qtype = ''
        if obl.entity and obl.entity.type == 'DATE':
            qtype = 'when'
            qword = 'Когда'
        elif obl.entity and obl.entity.type == 'LOC':
            qtype = 'where'
            qword = 'Где'
        elif len(obl._ents) == 1:
            ent: Entity = obl._ents[0]
            if ent.type == 'DATE':
                estart = ent.start
                estop = ent.stop
                aux_nodes = obl.contains(
                    lambda x: x.token.start > estop or x.token.stop < estart,
                    first=False
                )
                if len(aux_nodes) == 1 and aux_nodes[0].token.pos == 'ADP':  # TODO: add other adpos
                    qtype = 'when'
                    adp_token = aux_nodes[0].token
                    if adp_token.text.lower() == 'с':
                        verb_tense = root.token.feats.get('Tense')
                        if verb_tense in ['Past', 'Pres']:
                            qword = 'С какого момента'
                        else:
                            qword = 'Когда'
                    else:
                        qword = 'Когда'
            elif ent.type == 'LOC':
                qtype = 'where'
                estart = ent.start
                estop = ent.stop
                aux_nodes = obl.contains(
                    lambda x: x.token.start > estop or x.token.stop < estart,
                    first=False
                )
                if len(aux_nodes) == 1 and aux_nodes[0].token.pos == 'ADP':  # TODO: add other adpos
                    qtype = 'where'
                    adp_token = aux_nodes[0].token
                    if adp_token.text.lower() == 'с':
                        qword = 'С чего'
                    else:
                        qword = 'Где'
                else:
                    qword = 'Где'

        return qtype, qword


class AskNum(Ask):
    def __init__(self, root):
        super().__init__(root)
        self.numable = ('nummod', 'nummod:gov')

    @classmethod
    def qtype(self) -> str:
        return 'num'

    @classmethod
    def qwords(cls):
        return NUM_QWORDS

    def check(self, child):
        return child.contains(
            lambda x: x.type in self.numable
        ) is not None

    def generate(self, child, *args):
        nummod, _v = self.parse_answer_subtree(child)
        _q = self.get_qpart(child)
        q = tokens_to_text(_q, self.default_mapping())
        v = tokens_to_text(_v, self.default_mapping())
        q = f"Сколько {v} {q}"
        q = replace_punct(q)
        a = self.get_answer_subtree(nummod)
        return [QAPair(q, a, AskNum.qtype())]

    def parse_answer_subtree(self, child):
        nummod = child.contains(
            lambda x: x.type in self.numable
        )

        morph = MorphAnalyzer(lang='ru')
        new_obj = None
        try:
            new_obj = to_form(
                word=child.token.text,
                morph=morph,
                pos=child.token.pos,
                form={'gent', 'plur'}
            )
        except ValueError as e:
            print(e)
            print(child.token.pos)
        if new_obj is None:
            new_obj = child.token.text
        _v = collect_inorder(
            child,
            erase_ids={nummod.idx}
        )
        return nummod, _v

    def get_qpart(self, child):
        erase_ids = {
            x.idx for x in self.root.children
            if x.type in ['parataxis', 'advmod']
        } | {child.idx}
        type_order = sort_ids_by_typelist(
            ['root', 'nsubj', 'obj', 'obl']
        )

        _q = collect_inorder(
            self.root,
            erase_ids=erase_ids,
            order=type_order
        )
        return _q


class AskMoney(Ask):
    @classmethod
    def qtype(cls) -> str:
        return 'money'

    @classmethod
    def qword(cls):
        return ['']

    def check(self, child) -> bool:
        return False

    def generate(self, child, *args) -> list[QAPair]:
        return [QAPair('', '', '')]


class Generator(Ask):
    _qtypes = {AskNsubj: False,
               AskObj: False,
               AskObl: False}

    def __init__(self, root, _qtypes=None):
        super(Generator, self).__init__(root)

        if _qtypes is not None:
            self._qtypes = {
                _qt: _qtypes[_qt.qtype()]
                for _qt in self._qtypes
                if _qt.qtype() in _qtypes
            }
        self.qtypes = [c(root) for c in self._qtypes]

    def check(self, root):
        return None

    def generate(self):
        questions = []
        for c in self.root.children:
            for qtype in self.qtypes:
                if qtype.check(c):
                    all_opts = self._qtypes[type(qtype)]
                    qas = qtype.generate(c, all_opts)
                    aw = len(c._ents)
                    qw = len(self.root._ents) - aw
                    for qa in qas:
                        qa.qw = qw
                        qa.aw = aw
                    questions += qas
        return questions

    def generate_possible_questions(self, **kwargs):
        return self.generate(**kwargs)

    @classmethod
    def create_from_doc(cls, text, qtypes=None) -> Iterable['Generator']:
        doc = parse_to_doc(text)
        ents = Entity.parse_entities(doc)
        for sent in doc.sents:
            root = _Node.create_from_sentence(sent)
            if root is None or not root.validate(sent):
                continue
            root.sow_entities(ents)
            yield cls(root, qtypes)


if __name__ == '__main__':

    # # # Normal
    text = "В опросе приняло участие 2110 пользователей."
    # text = "Вес трона составляет 600 килограммов."
    # text = 'Один тиын составляет сотую часть тенге.'

    # # # Different question types
    # text = "Симма приговорили к двенадцати с половиной годам тюрьмы."
    # text = 'На сегодняшний день компания поставляет продукцию в 30 стран мира.'
    # text = 'Некоторых из них приговорили к 10-15 суткам ареста.'

    # # # Poor coordination
    # text = "Уже 4 лаборатории мира делали анализ ДНК."
    # text = 'С января по июль услугами «Трансаэро» воспользовались 52 человека.'

    # # # Auxiliary parts
    # text = "Общее число задержанных в ходе акций протеста с 15 июня по 6 июля составляет около 1800 человек."
    # text = "В Минске протестующих попросили прийти на площадь в 19 часов."

    # # Hard tree
    text = "Предыдущие учения с привлечением трех тысяч военнослужащих регулярной армии всех родов войск и резервистов Абхазия проводила в апреле."
    # text = 'Ранее в парламенте приводили данные, согласно которым 14 процентов браков в стране заключаются с несовершеннолетними девушками в возрасте 14-17 лет.'
    # text = 'Ежедневно мы несем убытки в миллионы долларов, около полумиллиона человек, занятых в главной отрасли молдавской экономики и связанных с ней других отраслях, остаются без работы.'
    # text = 'Молдавия предпримет еще одну попытку продать истребители МиГ-29, оставшиеся в республике с советских времен.'

    d = parse_to_doc(text, *default_parsers())
    ents = Entity.parse_entities(d)
    print(ents)
    sent = d.sents[0]

    # for t in sent.tokens:
    #     print(t)

    node = Generator.create_from_sentence(sent, ents)
    print(node.validate(sent))
    for vs in node.generate():
        print(vs)
        t, q, a = vs
        print(t, q)
        print(a)
