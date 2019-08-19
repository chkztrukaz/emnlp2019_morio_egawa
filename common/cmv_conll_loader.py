import os
from typing import List, Dict
import conllu as conllu

from . import io_util as io_util
from . import util as util
from . import data_class as DataClass


def __parse_sentence_lev(files: List[str]):
    result_docs = []
    post2filenames = {}
    for fname in files:
        basename = os.path.basename(fname)
        post_name = ''.join(basename.split('_')[:2])
        if post_name not in post2filenames:
            post2filenames[post_name] = []
        post2filenames[post_name].append(fname)

    for post_name, files in post2filenames.items():
        len_sentences = max(1, len(files) - 1)
        sorted_files = sorted(files, key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.dat', '')))
        for i, fname in enumerate(sorted_files):
            with open(fname, 'r') as content_file:
                content = content_file.read()
            chunks = conllu.parse(
                text=content,
            )
            assert len(chunks) == 1
            token_list, bio_list, tag_list, position_list, is_op_list = [], [], [], [], []
            for td in chunks[0]:
                bio_tags = td['xpostag'].split('-')
                tag = DataClass.ConllTag.NONE
                biostr = bio_tags[0]
                if biostr != 'O':
                    if 'Fact' in bio_tags[1]:
                        tag = DataClass.ConllTag.Fact
                    elif 'Testimony' in bio_tags[1]:
                        tag = DataClass.ConllTag.Testimony
                    elif 'Value' in bio_tags[1]:
                        tag = DataClass.ConllTag.Value
                    elif 'Rhetorical' in bio_tags[1]:
                        tag = DataClass.ConllTag.Rhetorical
                    elif 'Policy' in bio_tags[1]:
                        tag = DataClass.ConllTag.Policy
                    else:
                        assert False, 'Invalid proposition type: {}'.format(bio_tags[1])
                assert biostr in ['B', 'I', 'O']
                token = td['form']
                token_list.append(token)
                bio_list.append(biostr)
                tag_list.append(tag)
                position_list.append(i / len_sentences)
                is_op_list.append('_op' in fname)

            assert len(token_list) == len(bio_list) == len(tag_list)
            conll_doc = DataClass.ConllDocument(
                sentence_id=fname,
                token_list=token_list,
                bio_list=bio_list,
                tag_list=tag_list,
                option_features={
                    'position_list': position_list,
                    'is_op_list': is_op_list
                }
            )
            result_docs.append(conll_doc)
            if len(result_docs) % 100 == 0:
                util.print_info('loaded chunks: {}'.format(len(result_docs)))
    return result_docs


def __parse_post_lev(files: List[str]):
    result_docs = []
    for fname in files:
        with open(fname, 'r') as content_file:
            content = content_file.read()
        chunks = conllu.parse(
            text=content,
        )
        assert len(chunks) == 1
        token_list, bio_list, tag_list = [], [], []
        for td in chunks[0]:
            bio_tags = td['xpostag'].split('-')
            tag = DataClass.ConllTag.NONE
            biostr = bio_tags[0]
            if biostr != 'O':
                if 'Fact' in bio_tags[1]:
                    tag = DataClass.ConllTag.Fact
                elif 'Testimony' in bio_tags[1]:
                    tag = DataClass.ConllTag.Testimony
                elif 'Value' in bio_tags[1]:
                    tag = DataClass.ConllTag.Value
                elif 'Rhetorical' in bio_tags[1]:
                    tag = DataClass.ConllTag.Rhetorical
                elif 'Policy' in bio_tags[1]:
                    tag = DataClass.ConllTag.Policy
                else:
                    assert False, 'Invalid proposition type: {}'.format(bio_tags[1])
            assert biostr in ['B', 'I', 'O']
            token = td['form']
            token_list.append(token)
            bio_list.append(biostr)
            tag_list.append(tag)

        assert len(token_list) == len(bio_list) == len(tag_list)
        conll_doc = DataClass.ConllDocument(
            sentence_id=fname,
            token_list=token_list,
            bio_list=bio_list,
            tag_list=tag_list
        )
        result_docs.append(conll_doc)
        if len(result_docs) % 100 == 0:
            util.print_info('loaded chunks: {}'.format(len(result_docs)))
    return result_docs


def __load_conll(files: List[str], is_sentence_lev: bool):
    if is_sentence_lev:
        return __parse_sentence_lev(files=files)
    else:
        return __parse_post_lev(files=files)


def __load_cmv_conll(dir_conll: str, is_sentence_lev: bool):
    filenames = io_util.get_filenames_under_directory(path=dir_conll, extention='dat')
    filenames = [dir_conll + fname for fname in filenames]
    docs = __load_conll(
        filenames,
        is_sentence_lev=is_sentence_lev
    )
    return docs


def load_dataset(dir_conll: str, is_sentence_lev: bool) -> Dict:
    conll_docs = __load_cmv_conll(
        dir_conll=dir_conll,
        is_sentence_lev=is_sentence_lev
    )
    return {
        'documents': conll_docs,
        'tag2int': {
            DataClass.ConllTag.Fact: 5,
            DataClass.ConllTag.Testimony: 4,
            DataClass.ConllTag.Policy: 3,
            DataClass.ConllTag.Rhetorical: 2,
            DataClass.ConllTag.Value: 1,
            DataClass.ConllTag.NONE: 0
        }
    }


def main():
    return


if __name__ == '__main__':
    main()
