from typing import List, Dict
import random
import time
from subprocess import Popen, PIPE, STDOUT, DEVNULL
import copy
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.utils import shuffle as related_shuffle

from . import w2v_util as w2v_util
from . import data_class as DataClass
from . import config as cfg


def print_info(*text):
    if cfg.LOG_LEVEL <= 1:
        print(cfg.COL_INFO + '[ INFO ]', ' '.join(text), cfg.COL_END)


def print_ok(*text):
    if cfg.LOG_LEVEL <= 2:
        print(cfg.COL_OK + '[  OK  ]', ' '.join(text), cfg.COL_END)


def print_warn(*text):
    if cfg.LOG_LEVEL <= 3:
        print(cfg.COL_WARN + '[ WARN ]', ' '.join(text), cfg.COL_END)


def print_error(*text):
    if cfg.LOG_LEVEL <= 4:
        print(cfg.COL_FAIL + '[ERROR ]', ' '.join(text), cfg.COL_END)


def split_list(lst, trains: float = 0.8, samples: int = None):
    if samples is not None:
        index_samples = random.sample(range(len(lst)), samples)
        lst = [lst[i] for i in index_samples]
    sp = int(len(lst) * trains)
    train, test = lst[:sp], lst[sp:]
    return train, test


def get_precision_recall_f1_supp(y_true, y_pred, labels):
    ps, rs, f1s, sups = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, labels=labels)
    return ps, rs, f1s, sups


def get_f1_scores(y_true, y_pred, labels):
    return {
        'macro': f1_score(y_true, y_pred, labels=labels, average='macro'),
        'micro': f1_score(y_true, y_pred, labels=labels, average='micro'),
    }


def conll_bio2int(bio: str):
    """
    B = 2
    I = 1
    O = 0
    """
    if bio == 'B':
        return 2
    elif bio == 'I':
        return 1
    elif bio == 'O':
        return 0
    else:
        print_error('Could not encode {0}'.format(bio))


def make_conll_document_source_target_encode(
        documents: List[DataClass.ConllDocument],
        tag2int,
        max_length: int = None,
        use_option_features: bool = True
):
    """
    :param documents:
    :param w2index:
    :param sentence_pads:
    :return:
    """
    list_origin = []
    list_source = []
    list_tag = []
    list_bio = []
    for d in documents:
        if len(d.token_list) == 0:
            continue
        list_origin.append(d)
        embed = [w2v_util.w2v(t.lower())[0] for t in d.token_list]
        bios = [conll_bio2int(bio) for bio in d.bio_list]
        tags = [tag2int[tag] for tag in d.tag_list]

        if use_option_features and d.option_features:
            additional_feature = []
            sorted_features = [v for (k, v) in sorted(d.option_features.items(), key=lambda x: x[0])]
            for feat_tpl in zip(*sorted_features):
                additional_feature.append(np.array(feat_tpl, dtype='f'))
            assert len(embed) == len(additional_feature)
            embed = [np.concatenate([embed, additional], 0) for embed, additional in zip(embed, additional_feature)]

        if max_length is not None and max_length < len(embed):
            print_warn('Cut too long tokens: {} -> {}'.format(len(embed), max_length))
            embed = embed[:max_length]
            tags = tags[:max_length]
            bios = bios[:max_length]

        list_source.append(
            embed
        )
        list_tag.append(
            tags
        )
        list_bio.append(
            bios
        )

    assert len(list_source) == len(list_tag) == len(list_bio)

    def zipdata(data_dict):
        data_out = []
        data_tpl = data_dict.items()
        data_keys = [k for k, v in data_tpl]
        data_lsts = [v for k, v in data_tpl]
        for dlst in zip(*data_lsts):
            data_out.append(
                {
                    k: dlst[i] for i, k in enumerate(data_keys)
                }
            )
        return data_out

    return zipdata({
        'origin': list_origin,
        'source': list_source,
        'tag': list_tag,
        'bio': list_bio,
    })


def get_conll_documents_train_test_validation(
        documents: List[DataClass.ConllDocument],
        tag2int: Dict[DataClass.ConllTag, int] = None,
        max_length: int = None,
        use_option_features: bool = True,
        train_rate=.8,
        dev_rate=.3,
        rnd_state=None
):
    def _sente2doc_id(sentence_id: str):
        return os.path.basename(sentence_id)[:3]

    doc_ids = list(set([_sente2doc_id(d.sentence_id) for d in documents]))
    doc_ids = related_shuffle(doc_ids, random_state=rnd_state)
    n_train = int(len(doc_ids) * train_rate)
    n_dev = int(n_train * dev_rate)
    train_doc_ids = doc_ids[n_dev: n_train]
    dev_doc_ids = doc_ids[:n_dev]
    test_doc_ids = doc_ids[n_train:]

    trains = [d for d in documents if _sente2doc_id(d.sentence_id) in train_doc_ids]
    tests = [d for d in documents if _sente2doc_id(d.sentence_id) in test_doc_ids]
    devs = [d for d in documents if _sente2doc_id(d.sentence_id) in dev_doc_ids]

    train_data = make_conll_document_source_target_encode(
        documents=trains,
        tag2int=tag2int,
        max_length=max_length,
        use_option_features=use_option_features
    )
    test_data = make_conll_document_source_target_encode(
        documents=tests,
        tag2int=tag2int,
        max_length=max_length,
        use_option_features=use_option_features
    )
    dev_data = make_conll_document_source_target_encode(
        documents=devs,
        tag2int=tag2int,
        max_length=max_length,
        use_option_features=use_option_features
    )
    return {
        'train': train_data,
        'test': test_data,
        'dev': dev_data,
    }


def run_procs_using_gpu(py_processes: List[str], available_gpus: List[int], max_proc_per_gpu: int):
    running_procs = []
    py_processes = copy.deepcopy(py_processes)
    while py_processes or running_procs:
        if py_processes and len(running_procs) < len(available_gpus) * max_proc_per_gpu:
            # Find available GPUs
            gpu_assign = None
            for gpu_cand in available_gpus:
                if len([True for rproc in running_procs if rproc['gpu'] == gpu_cand]) < max_proc_per_gpu:
                    gpu_assign = gpu_cand
                    break
            assert gpu_assign is not None
            # Run processes
            args = py_processes.pop(0)
            str_args = [arg for arg in 'python {0} --gpu {1}'.format(args, gpu_assign).split(' ')
                        if arg != '']
            proc = Popen(str_args, stdout=DEVNULL, stderr=DEVNULL)
            running_procs.append(
                {
                    'gpu': gpu_assign,
                    'proc': proc
                }
            )
            print_info('running (gpu:{0}, pid:{1}) {2}'.format(gpu_assign, proc.pid, args))
        for rproc in running_procs:
            retcode = rproc['proc'].poll()
            if retcode is not None:  # Process finished.
                running_procs.remove(rproc)
                print_ok('finished (gpu:{0}) {1}'.format(rproc['gpu'], rproc['proc']))
                break
            else:  # No process is done, wait a bit and check again.
                continue
        time.sleep(0.5)
    return
