# coding: utf-8

from typing import List
import pandas as pd
import argparse

import chainer

from common import gpu_config as gpu_config
from common import data_class as DataClass
from common import util as util
from common import w2v_util as w2v_util
from model.BLC import BLC, batch_convert


test_documents = [
    ['hello', '.'],
]

parser = argparse.ArgumentParser(description=u'')
parser.add_argument('--glove', type=str, default='./dataset/glove.6B.100d.gz', help=u'Glove embedding path.')
parser.add_argument('--model_file', type=str, default='./trained_model/best_model', help=u'The trained model file.')
parser.add_argument('--param_file', type=str, default='./trained_model/param.csv', help=u'The parameter file.')
args = parser.parse_args()

PARAM_GLOVE_PATH = args.glove
MODEL_FILE = args.model_file
PARAM_FILE = args.param_file


def load_model(
        model_file: str,
        n_units: int,
        in_size: int,
        out_size_bio: int,
        out_size_tag: int,
        blstm_stack: int,
        lossfun: str,
        gpu: int = -1
):
    model = BLC(
        n_units=n_units,
        in_size=in_size,
        n_outs_bio=out_size_bio,
        n_outs_tag=out_size_tag,
        blstm_stack=blstm_stack,
        lossfun=lossfun,
        dropout=0,
        weight_bio=0.5,
        weight_tag=0.5
    )
    chainer.serializers.load_npz(model_file, model)
    if gpu_config.disable_gpu or gpu < 0:
        model.to_cpu()
    else:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu(gpu)
    return model


def correct_bios_tags(bios, tags):
    assert len(bios) == len(tags)
    need_recursive_correction = False
    new_bios, new_tags = [], []
    for i, (b, t) in enumerate(zip(bios, tags)):
        new_tag = t
        new_bio = b
        if b == 'I' and i != 0:
            if bios[i - 1] == 'O':
                new_bio = 'B'
        if b == 'O' and t != DataClass.ConllTag.NONE:
            new_tag = DataClass.ConllTag.NONE
        if b in ['B', 'I'] and t == DataClass.ConllTag.NONE:
            new_bio = 'O'
            need_recursive_correction = True

        new_bios.append(new_bio)
        new_tags.append(new_tag)

    if need_recursive_correction:
        new_bios, new_tags = correct_bios_tags(new_bios, new_tags)

    return new_bios, new_tags


def test(model, documents: List[DataClass.ConllDocument], tag2int, batch_size: int, gpu: int):
    int2tag = {i: t for t, i in tag2int.items()}
    str_bio = ['B', 'I', 'O']
    int2bio = {i: s for s, i in zip(str_bio, [util.conll_bio2int(s) for s in str_bio])}
    test_data = util.make_conll_document_source_target_encode(
        documents=documents,
        tag2int=tag2int,
        max_length=None,
    )

    result_bios = []
    result_tags = []
    for i in range(0, len(test_data), batch_size):
        test_batch = test_data[i:i + batch_size]
        converted_data = batch_convert(test_batch, gpu)
        predict_bios, predict_tags = model.test(
            source=converted_data['source'],
        )
        result_bios += [[int2bio[i] for i in pb] for pb in predict_bios]
        result_tags += [[int2tag[i] for i in pt] for pt in predict_tags]
    return {
        'bio_list': result_bios,
        'tag_list': result_tags
    }


def main():
    # Load GloVe embeddings
    util.print_info('Loading GloVe embeddings.')
    w2v_util.load_embedding_dict(model_path=PARAM_GLOVE_PATH, normalize_digits=True)
    # Special symbol dictionary
    spec_w2vec = w2v_util.create_spec_w2vec(n_dim=w2v_util.embedd_dim)
    # Add the special dictionary to the GloVe embeddings
    w2v_util.embedd_dict.update(spec_w2vec)

    gpu = -1
    param2value = pd.read_csv(filepath_or_buffer=PARAM_FILE, sep=',', header=None, index_col=0).to_dict()[1]
    model = load_model(
        model_file=MODEL_FILE,
        n_units=int(param2value['units']),
        in_size=int(param2value['in_size']),
        out_size_bio=int(param2value['out_size_bio']),
        out_size_tag=int(param2value['out_size_tag']),
        blstm_stack=int(param2value['BiLSTM stack']),
        lossfun=param2value['loss_function'],
        gpu=gpu
    )

    global test_documents
    test_conll_documents = [
            DataClass.ConllDocument(
            sentence_id='0',
            token_list=tokens,
            bio_list=['O' for _ in tokens],  # Dummy
            tag_list=[DataClass.ConllTag.NONE for _ in tokens]  # Dummy
        )
        for tokens in test_documents]
    result = test(
        model=model,
        documents=test_conll_documents,
        tag2int={
            DataClass.ConllTag.Fact: 5,
            DataClass.ConllTag.Testimony: 4,
            DataClass.ConllTag.Policy: 3,
            DataClass.ConllTag.Rhetorical: 2,
            DataClass.ConllTag.Value: 1,
            DataClass.ConllTag.NONE: 0
        },
        batch_size=16,
        gpu=gpu
    )
    for bio, tag, tokens in zip(result['bio_list'], result['tag_list'], test_documents):
        bio, tag = correct_bios_tags(bios=bio, tags=tag)

        stock_tokens = []
        is_in_boundary=False
        for i, (b, t, token) in enumerate(zip(bio, tag, tokens)):
            if b == 'B':
                stock_tokens.append('[')
                is_in_boundary = True
            if b == 'O' and is_in_boundary:
                stock_tokens.append(']({})'.format(str(tag[i - 1]).split('.')[-1]))
                is_in_boundary = False
            stock_tokens.append(token)
        print(' '.join(stock_tokens))


if __name__ == '__main__':
    main()
