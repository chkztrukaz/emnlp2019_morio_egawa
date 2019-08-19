import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import cuda, training, reporter


class BLC(chainer.Chain):
    """
    BiLSTM CRF
    """

    def __init__(
            self,
            n_units: int,
            in_size: int,
            n_outs_bio: int,
            n_outs_tag: int,
            blstm_stack: int = 1,
            lossfun='crf',
            dropout=0.2,
            weight_bio=0.5,
            weight_tag=0.5
    ):
        assert n_units > 0
        assert n_outs_bio > 0
        assert n_outs_tag > 0
        assert blstm_stack >= 0
        assert 0 <= dropout <= 1
        assert weight_bio >= 0 and weight_tag >= 0
        assert weight_bio + weight_tag <= 1.0
        self.blstm_stack = blstm_stack
        self.inds = None
        self.xs_src_len = None
        self.lossfun = lossfun
        self.dropout = dropout
        self.weight_bio = weight_bio
        self.weight_tag = weight_tag
        super(BLC, self).__init__()
        with self.init_scope():
            # Stack BiLSTM
            if blstm_stack > 0:
                self.bilstm = L.NStepBiLSTM(
                    n_layers=blstm_stack,
                    in_size=in_size,
                    out_size=n_units,
                    dropout=self.dropout
                )
            self.out_layer_bio = L.Linear(None, n_outs_bio)
            self.out_layer_tag = L.Linear(None, n_outs_tag)
            # CRF layer
            if lossfun == 'crf':
                self.crf_bio = L.CRF1d(n_outs_bio)
                self.crf_tag = L.CRF1d(n_outs_tag)

    def forward(self, source):
        """
        Propagate according to the network and output for each timestep in the BiLSTM
        :param source: (sequence_length, n_embed) x batch_size
        :return:
        """
        ys = []
        for s in source:
            ys.append(
                F.stack(s)
            )

        # BiLSTM
        if self.blstm_stack > 0:
            # ys: (sequence_length, n_units) x batch_size
            _, _, ys = self.bilstm(
                hx=None, cx=None, xs=ys
            )  # batch_size x (sequence_length, 2 * n_units)

        ys_flat = F.concat(ys, axis=0)  # (batch_size * sequence_length, hidden)

        # Label Predict
        o_flat_bio = self.out_layer_bio(ys_flat)  # (batch_size * sequence_length, n_outs_bio)
        o_flat_tag = self.out_layer_tag(ys_flat)  # (batch_size * sequence_length, n_outs_tag)

        o_list_bio = F.split_axis(o_flat_bio, o_flat_bio.data.shape[0], axis=0)
        # -> {batch_size * sequence_length} x (1, n_outs_bio)
        o_list_tag = F.split_axis(o_flat_tag, o_flat_tag.data.shape[0], axis=0)
        # -> {batch_size * sequence_length} x (1, n_outs_tag)

        pred_list_bio = []
        pred_list_tag = []
        cnt = 0
        for n_len in self.xs_src_len:
            # BIO
            pred_bio = F.concat(o_list_bio[cnt:cnt + n_len], axis=0)
            pred_list_bio.append(pred_bio)
            # tag
            pred_tag = F.concat(o_list_tag[cnt:cnt + n_len], axis=0)
            pred_list_tag.append(pred_tag)
            cnt += n_len
        # pred_list_bio -> batch_size x (sequence_length, n_outs_bio)
        # pred_list_tag -> batch_size x (sequence_length, n_outs_tag)
        return pred_list_bio, pred_list_tag

    def __call__(self, source, bio, tag, compute_loss=True):
        """
        Conduct forward propagation and acquire the loss value
        :return: loss (a chainer variable)
        """
        # Order by a sequence length
        self.inds = np.argsort([-len(x) for x in source]).astype('i')  # Remember the original order
        xs_src = [source[i] for i in self.inds]
        self.xs_src_len = [len(x) for x in xs_src]  # Remember the batch length

        # Forward propagation
        pred_list_bio, pred_list_tag = self.forward(
            source=xs_src,
        )  # batch_size x (sequence_length, 2 * n_units)

        # Calculate the loss
        loss_bio = chainer.Variable(self.xp.array(0, dtype='f'))
        loss_tag = chainer.Variable(self.xp.array(0, dtype='f'))
        # Predict the outputs
        predicts_bio = []
        predicts_tag = []
        # If we use CRFs as output layers
        if self.lossfun == 'crf':
            # ------------------
            # bio
            # ------------------
            hs_bio = F.transpose_sequence(pred_list_bio)  # sequence_length x (batch_size)
            # Loop for each batch and get loss values
            if compute_loss:
                ys_bio = [bio[i] for i in self.inds]
                ts_bio = F.transpose_sequence(ys_bio)  # sequence_length x (batch_size)
                loss_bio = self.crf_bio(hs_bio, ts_bio)
            # Add prediction results
            _, predicts_trans_bio = self.crf_bio.argmax(hs_bio)
            predicts_bio = F.transpose_sequence(predicts_trans_bio)
            # ------------------
            # bio
            # ------------------
            hs_tag = F.transpose_sequence(pred_list_tag)  # sequence_length x (batch_size)
            # Loop for each batch and get loss values
            if compute_loss:
                ys_tag = [tag[i] for i in self.inds]
                ts_tag = F.transpose_sequence(ys_tag)  # sequence_length x (batch_size)
                loss_tag = self.crf_tag(hs_tag, ts_tag)
            # Add prediction results
            _, predicts_trans_tag = self.crf_tag.argmax(hs_tag)
            predicts_tag = F.transpose_sequence(predicts_trans_tag)
        elif self.lossfun == 'softmax':
            # ------------------
            # bio
            # ------------------
            if compute_loss:
                ys_bio = [bio[i] for i in self.inds]
                # Loop for each batch and get loss values
                for p_lst, y_lst in zip(pred_list_bio, ys_bio):
                    loss_bio += F.softmax_cross_entropy(p_lst, y_lst)
                loss_bio /= len(bio)
            # Add prediction results
            for p_lst in pred_list_bio:
                y_arg_bio = F.argmax(p_lst, axis=1)
                predicts_bio.append(y_arg_bio)
            # ------------------
            # tag
            # ------------------
            if compute_loss:
                ys_tag = [tag[i] for i in self.inds]
                # Loop for each batch and get loss values
                for p_lst, y_lst in zip(pred_list_tag, ys_tag):
                    loss_tag += F.softmax_cross_entropy(p_lst, y_lst)
                loss_tag /= len(tag)
            # Add prediction results
            for p_lst in pred_list_tag:
                y_arg_tag = F.argmax(p_lst, axis=1)
                predicts_tag.append(y_arg_tag)

        # Transform variable from GPU to CPU
        cpu_predicts_bio = []
        cpu_predicts_tag = []
        for pred_bio, pred_tag in zip(predicts_bio, predicts_tag):
            cpu_predicts_bio.append(chainer.cuda.to_cpu(pred_bio.data).tolist())
            cpu_predicts_tag.append(chainer.cuda.to_cpu(pred_tag.data).tolist())
        # Re-order
        inds_rev = sorted([(i, ind) for i, ind in enumerate(self.inds)], key=lambda x: x[1])
        cpu_predicts_bio = [cpu_predicts_bio[e_i] for e_i, _ in inds_rev]
        cpu_predicts_tag = [cpu_predicts_tag[e_i] for e_i, _ in inds_rev]

        if compute_loss:
            loss = self.weight_bio * loss_bio + self.weight_tag * loss_tag
            return loss, cpu_predicts_bio, cpu_predicts_tag
        else:
            return cpu_predicts_bio, cpu_predicts_tag

    def test(self, source):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            cpu_predicts_bio, cpu_predicts_tag = self.__call__(
                source=source,
                bio=None,
                tag=None,
                compute_loss=False,
            )
            return cpu_predicts_bio, cpu_predicts_tag


def batch_convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {
        'source': to_device_batch(
            [np.array(b['source'], dtype='f') for b in batch]
        ),
        'tag': to_device_batch(
            [np.array(b['tag'], dtype='i') for b in batch]
        ),
        'bio': to_device_batch(
            [np.array(b['bio'], dtype='i') for b in batch]
        )
    }


class BLCUpdater(training.StandardUpdater):
    def __init__(self, train_iterator, model: BLC, optimizer,
                 device=None):
        iterator = {'main': train_iterator}
        self._iterators = iterator
        self.model = model
        self._optimizers = {'main': optimizer}
        self.converter = batch_convert
        self.device = device
        self.iteration = 0

    def update_core(self):
        iterator = self._iterators['main'].next()
        in_arrays = self.converter(iterator, self.device)
        loss, _, _ = self.model(
            source=in_arrays['source'],
            bio=in_arrays['bio'],
            tag=in_arrays['tag'],
            compute_loss=True
        )
        self._optimizers['main'].target.cleargrads()
        reporter.report({'loss': loss}, self.model)
        loss.backward()
        self._optimizers['main'].update()

