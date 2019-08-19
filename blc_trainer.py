# coding: utf-8

import argparse
import random
from typing import List
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

from common import gpu_config as gpu_config
from common import data_class as DataClass
from common import cmv_conll_loader as data_loader
from common import util as util
from common import w2v_util as w2v_util
from common import io_util as io_util
from common.chainer_save_restore import SaveRestore

from model.BLC import BLC, BLCUpdater, batch_convert


TRAIN_LOG = 'train.txt'
PARAM_LOG = 'param.csv'
TEST_LOG = 'test.txt'
EVAL_LOG = 'eval.csv'

parser = argparse.ArgumentParser(description=u'')
parser.add_argument('--glove', type=str, default='./dataset/glove.6B.100d.gz', help=u'Glove embedding path.')
parser.add_argument('--sentence_level', type=int, default=1, help=u'Sentence level or post level input.')
parser.add_argument('--use_option_features', type=int, default=1, help=u'Use optional input if SENTENCE_LEVEL')

parser.add_argument('--batch', type=int, default=16, help=u'Batch size')
parser.add_argument('--units', type=int, default=100, help=u'Unit size')
parser.add_argument('--dropout', type=float, default=0.33, help=u'Dropout rate of LSTMs')
parser.add_argument('--blstm', type=int, default=1, help=u'BiLSTM stack size')
parser.add_argument('--lossfun', type=str, default='crf', help=u'Loss function (crf or softmax)')
parser.add_argument('--max_length', type=int, default=None, help=u'Max sequence size')
parser.add_argument('--epoch', type=int, default=50, help=u'Epoch num')
parser.add_argument('--gpu', type=int, default=gpu_config.GPU, help=u'GPU number')
parser.add_argument('--logdir', type=str, default='log/blc_log', help=u'Log directory')
parser.add_argument('--self_train', type=int, default=0, help=u'Train with full dataset')
parser.add_argument('--dev_rate', type=float, default=0.3, help=u'Development set ratio')
parser.add_argument('--only_bio', type=int, default=0, help=u'Train only BIO output')
parser.add_argument('--only_tag', type=int, default=0, help=u'Train only tag output')
args = parser.parse_args()

PARAM_GLOVE_PATH = args.glove

PARAM_USE_OPTION_FEATURES = args.use_option_features > 0

PARAM_SENTENCE_LEVEL = args.sentence_level > 0
if PARAM_SENTENCE_LEVEL:
    PARAM_TRAIN_CONLL = './dataset/conll_format/sentence_level/'
else:
    PARAM_TRAIN_CONLL = './dataset/conll_format/post_level/'

PARAM_BATCH_SIZE = args.batch
assert PARAM_BATCH_SIZE > 0

PARAM_N_UNITS = args.units
assert PARAM_N_UNITS > 0

PARAM_DROPOUT = args.dropout
assert 1 >= PARAM_DROPOUT >= 0

PARAM_BLSTM = args.blstm
assert 5 > PARAM_BLSTM >= 0

PARAM_LOSSFUN = args.lossfun
assert PARAM_LOSSFUN in ['crf', 'softmax']

PARAM_MAX_LENGTH = args.max_length
assert PARAM_MAX_LENGTH is None or PARAM_MAX_LENGTH > 0

PARAM_EPOCH = args.epoch
assert PARAM_EPOCH > 0

PARAM_GPU = args.gpu
assert PARAM_GPU >= -1

PARAM_SELF_TRAIN = args.self_train > 0

PARAM_DEV_RATE = args.dev_rate
assert 1.0 > PARAM_DEV_RATE > 0.0

PARAM_ONLY_BIO = args.only_bio > 0
PARAM_ONLY_TAG = args.only_tag > 0

PARAM_LOGDIR = args.logdir


def create_model(in_size: int, out_size_bio: int, out_size_tag: int):
    model = BLC(
        n_units=PARAM_N_UNITS,
        in_size=in_size,
        n_outs_bio=out_size_bio,
        n_outs_tag=out_size_tag,
        blstm_stack=PARAM_BLSTM,
        lossfun=PARAM_LOSSFUN,
        dropout=PARAM_DROPOUT,
        weight_bio=0 if PARAM_ONLY_TAG else 0.5,
        weight_tag=0 if PARAM_ONLY_BIO else 0.5
    )
    if gpu_config.disable_gpu:
        model.to_cpu()
    else:
        chainer.cuda.get_device_from_id(PARAM_GPU).use()
        model.to_gpu(PARAM_GPU)
    return model


def evaluate(model, data, batch_size, tag2int, device):
    yts_bio, yps_bio = [], []  # BIO
    yts_tag, yps_tag = [], []  # tag
    for i in range(0, len(data), batch_size):
        test_batch = data[i:i + batch_size]
        true_bios = [b['bio'] for b in test_batch]
        true_tags = [b['tag'] for b in test_batch]
        converted_data = batch_convert(test_batch, device)
        predict_bios, predict_tags = model.test(
            source=converted_data['source'],
        )
        # BIO
        for yts, yps in zip(true_bios, predict_bios):
            assert len(yts) == len(yps)
            yts_bio += yts
            yps_bio += yps
        # tag
        for yts, yps in zip(true_tags, predict_tags):
            assert len(yts) == len(yps)
            yts_tag += yts
            yps_tag += yps
    labels_bio = [
        util.conll_bio2int(bio='B'),
        util.conll_bio2int(bio='I'),
        util.conll_bio2int(bio='O'),
    ]
    sorted_tag2int = sorted(tag2int.items(), key=lambda x: x[1])
    labels_tag = [
        i for t, i in sorted_tag2int
    ]

    # BIO
    ps_bio, rs_bio, fs_bio, _ = util.get_precision_recall_f1_supp(
        y_true=yts_bio,
        y_pred=yps_bio,
        labels=labels_bio
    )
    fscore_dict_bio = util.get_f1_scores(
        y_true=yts_bio,
        y_pred=yps_bio,
        labels=labels_bio
    )

    #tag
    ps_tag, rs_tag, fs_tag, _ = util.get_precision_recall_f1_supp(
        y_true=yts_tag,
        y_pred=yps_tag,
        labels=labels_tag
    )
    fscore_dict_tag = util.get_f1_scores(
        y_true=yts_tag,
        y_pred=yps_tag,
        labels=labels_tag
    )
    return {
        'bio': (ps_bio, rs_bio, fs_bio, fscore_dict_bio),
        'tag': (ps_tag, rs_tag, fs_tag, fscore_dict_tag)
    }


def train(documents: List[DataClass.ConllDocument], tag2int):
    PARAM_LOGGER = io_util.get_logger(PARAM_LOGDIR + '/' + PARAM_LOG, clear=True)
    TEST_LOGGER = io_util.get_logger(PARAM_LOGDIR + '/' + TEST_LOG, clear=True)
    EVAL_LOGGER = io_util.get_logger(PARAM_LOGDIR + '/' + EVAL_LOG, clear=True)

    util.print_info('Using GPU No.: {}'.format(PARAM_GPU))
    util.print_info('log_dir: {}'.format(PARAM_LOGDIR))

    PARAM_LOGGER('batch_size,{}'.format(PARAM_BATCH_SIZE))
    PARAM_LOGGER('units,{}'.format(PARAM_N_UNITS))
    PARAM_LOGGER('dropout,{}'.format(PARAM_DROPOUT))
    PARAM_LOGGER('gpu,{}'.format(PARAM_GPU))
    PARAM_LOGGER('BiLSTM stack,{}'.format(PARAM_BLSTM))
    PARAM_LOGGER('loss_function,{}'.format(PARAM_LOSSFUN))
    PARAM_LOGGER('epoch,{}'.format(PARAM_EPOCH))
    PARAM_LOGGER('max_length,{}'.format(PARAM_MAX_LENGTH))
    PARAM_LOGGER('self_train,{}'.format(PARAM_SELF_TRAIN))
    PARAM_LOGGER('only_bio,{}'.format(PARAM_ONLY_BIO))
    PARAM_LOGGER('only_tag,{}'.format(PARAM_ONLY_TAG))
    PARAM_LOGGER('use_option_features,{}'.format(PARAM_USE_OPTION_FEATURES))

    data_dict = util.get_conll_documents_train_test_validation(
        documents=documents,
        tag2int=tag2int,
        max_length=PARAM_MAX_LENGTH,
        use_option_features=PARAM_USE_OPTION_FEATURES,
        train_rate=1 if PARAM_SELF_TRAIN else .8,
        dev_rate=PARAM_DEV_RATE,
        rnd_state=None
    )
    train_data = data_dict['train']
    test_data = data_dict['test']
    valid_data = data_dict['dev']

    in_size = len(train_data[0]['source'][0])

    PARAM_LOGGER('in_size,%d' % in_size)
    PARAM_LOGGER('train_data,%d' % len(train_data))
    PARAM_LOGGER('valid_data,%d' % len(valid_data))
    PARAM_LOGGER('test_data,%d' % len(test_data))
    PARAM_LOGGER('out_size_bio,%d' % 3)
    PARAM_LOGGER('out_size_tag,%d' % len(tag2int))

    model = create_model(in_size=in_size, out_size_bio=3, out_size_tag=len(tag2int))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, PARAM_BATCH_SIZE)
    updater = BLCUpdater(
        train_iterator=train_iter,
        model=model,
        optimizer=optimizer,
        device=PARAM_GPU
    )
    trainer = training.Trainer(updater, (PARAM_EPOCH, 'epoch'), out=PARAM_LOGDIR)
    trainer.extend(extensions.LogReport(trigger=(5, 'iteration'), log_name=TRAIN_LOG))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'elapsed_time']),
        trigger=(5, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    sorted_tag2int = sorted(tag2int.items(), key=lambda x: x[1])

    def record_scores(trainer):
        eval_dict = evaluate(
            model=model,
            data=test_data,
            batch_size=128,
            tag2int=tag2int,
            device=updater.device
        )
        ps_bio, rs_bio, fs_bio, fscore_dict_bio = eval_dict['bio']
        ps_tag, rs_tag, fs_tag, fscore_dict_tag = eval_dict['tag']

        def round(np_scalar, size=3):
            return np.round(np_scalar, size)
        csv_out = [
            trainer.updater.epoch,
            trainer.updater.iteration,
            round(ps_bio[0], size=4), round(rs_bio[0], size=4), round(fs_bio[0], size=4),
            round(ps_bio[1], size=4), round(rs_bio[1], size=4), round(fs_bio[1], size=4),
            round(ps_bio[2], size=4), round(rs_bio[2], size=4), round(fs_bio[2], size=4),
            round(fscore_dict_bio['micro'], size=4),
            round(fscore_dict_bio['macro'], size=4),
        ]
        for t, i in sorted_tag2int:
            csv_out.append(round(ps_tag[i], size=4))
            csv_out.append(round(rs_tag[i], size=4))
            csv_out.append(round(fs_tag[i], size=4))
        csv_out += [
            round(fscore_dict_tag['micro'], size=4),
            round(fscore_dict_tag['macro'], size=4),
        ]
        formstr = ''.join('{},' * len(csv_out))[:-1]
        EVAL_LOGGER(formstr.format(*csv_out))
    EVAL_LOGGER(
        'epoch,' +
        'iter,' +
        'B-prec,B-rec,B-f1,' +
        'I-prec,I-rec,I-f1,' +
        'O-prec,O-rec,O-f1,' +
        'BIO-micro-f1,' +
        'BIO-macro-f1,' +
        ''.join(['{0}-prec,{0}-rec,{0}-f1,'.format(t) for t, i in sorted_tag2int]) +
        'tag-micro-f1,' +
        'tag-macro-f1'
    )

    valid_saver = SaveRestore(filename=None)
    current_validation_f1 = -float('inf')

    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def validation(trainer):
        nonlocal valid_saver
        nonlocal current_validation_f1
        eval_dict = evaluate(
            model=model,
            data=valid_data,
            batch_size=64,
            tag2int=tag2int,
            device=updater.device
        )
        if PARAM_ONLY_TAG:
            ps_tag, rs_tag, fs_tag, fscore_dict_tag = eval_dict['tag']
            new_validation_f1 = fscore_dict_tag['macro']
        else:
            ps_bio, rs_bio, fs_bio, fscore_dict_bio = eval_dict['bio']
            new_validation_f1 = fscore_dict_bio['macro']
        if new_validation_f1 > current_validation_f1:
            current_validation_f1 = new_validation_f1
            if PARAM_SELF_TRAIN:
                valid_saver(trainer=trainer)
                chainer.serializers.save_npz(PARAM_LOGDIR + '/best_model', model)
                util.print_info('Saved best trainer, model/ epoch: {}, iter:{}'.format(
                    trainer.updater.epoch,
                    trainer.updater.iteration,
                ))
            else:
                record_scores(trainer=trainer)
                util.print_info('Recorded scores/ epoch: {}, iter:{}'.format(
                    trainer.updater.epoch,
                    trainer.updater.iteration,
                ))
    trainer.extend(validation)

    @chainer.training.make_extension(trigger=(50, 'iteration'))
    def test(trainer):
        if PARAM_SELF_TRAIN:
            return
        TEST_LOGGER('-- iter: {0} --'.format(trainer.updater.iteration))
        for _ in range(3):
            test_trg = random.choice(test_data)
            true_bios = test_trg['bio']
            true_tags = test_trg['tag']
            converted_data = batch_convert([test_trg], updater.device)
            predict_bios, predict_tags = model.test(
                source=converted_data['source'],
            )
            predict_bios = predict_bios[0]
            predict_tags = predict_tags[0]
            assert len(true_bios) == len(predict_bios) == len(true_tags) == len(predict_tags)
            TEST_LOGGER('bio y: {0}'.format(true_bios))
            TEST_LOGGER('bio ȳ: {0}'.format(predict_bios))
            TEST_LOGGER('tag y: {0}'.format(true_tags))
            TEST_LOGGER('tag ȳ: {0}'.format(predict_tags))
            TEST_LOGGER('')
    trainer.extend(test)

    trainer.run()
    PARAM_LOGGER('train_finished,{}'.format(True))
    return


def main():
    # Load GloVe embeddings
    util.print_info('Loading GloVe embeddings.')
    w2v_util.load_embedding_dict(model_path=PARAM_GLOVE_PATH, normalize_digits=True)
    # Special symbol dictionary
    spec_w2vec = w2v_util.create_spec_w2vec(n_dim=w2v_util.embedd_dim)
    # Add the special dictionary to the GloVe embeddings
    w2v_util.embedd_dict.update(spec_w2vec)

    # Load dataset
    util.print_info('Loading dataset.')
    dataset = data_loader.load_dataset(
        dir_conll=PARAM_TRAIN_CONLL,
        is_sentence_lev=PARAM_SENTENCE_LEVEL
    )

    # Conduct train
    train(
        documents=dataset['documents'],
        tag2int=dataset['tag2int']
    )


if __name__ == '__main__':
    main()
