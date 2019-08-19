import enum
from typing import List, Dict


class ConllTag(enum.Enum):
    """
    Proposition tag
    """
    NONE = 0
    Value = 1
    Policy = 2
    Rhetorical = 3
    Fact = 4
    Testimony = 5


class ConllDocument:
    def __init__(self,
                 sentence_id: str,
                 token_list: List[str],
                 bio_list: List[str],
                 tag_list: List[ConllTag],
                 option_features: Dict=None,
                 ):
        self.sentence_id = sentence_id
        self.token_list = token_list
        self.tag_list = tag_list
        self.bio_list = bio_list
        self.option_features = option_features

        assert len(token_list) == len(bio_list) == len(tag_list)
        return

    def call(self):
        return

