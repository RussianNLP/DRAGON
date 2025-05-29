from QAGenerator import Ask
from src.utils import collect_inorder


class AskAbout(Ask):
    @classmethod
    def qtype(cls) -> str:
        return 'about'

    @classmethod
    def qwords(cls):
        return 'О чем'

    def check(self, child) -> bool:
        lemmas = {
            t.lemma for t in collect_inorder(child)
        }
        return child.type == 'obl' and lemmas == {'о', 'это'}

    def generate(self, child, all_opts=False):
        erase_ids = {
            x.idx for x in self.root.children
            if x.type in ['parataxis', 'advmod']
        }
        erase_ids.add(child.idx)


# class GeneratorByDoc:
#     def __init__(self):
#         pass

#     @classmethod
#     def create_from_doc(cls, text, qtypes):
