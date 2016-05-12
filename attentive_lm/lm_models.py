"""
"""

from __future__ import print_function
import time
import random
import numpy
import tensorflow as tf
from tensorflow.models.rnn import seq2seq, rnn
from tensorflow.python.ops import array_ops, math_ops, nn_ops

import cells
import data_utils
import lm_ops
import optimization_ops


_SEED = 1234


class LMModel(object):
    """The PTB model."""

    def __init__(self,
                 is_training,
                 learning_rate=1.0,
                 optimizer="sgd",
                 max_grad_norm=5,
                 num_layers=2,
                 use_lstm=True,
                 num_steps=35,
                 num_steps_valid=120,
                 proj_size=650,
                 hidden_size=650,
                 hidden_proj=650,
                 num_samples=512,
                 init_scale=0.1,
                 dropout_rate=0.0,
                 lr_decay=0.8,
                 batch_size=20,
                 attentive=False,
                 projection_attention_f=None,
                 output_form=lm_ops.OUTPUT_CONCAT,
                 vocab_size=10000):

        with tf.device("/gpu:0"):

            if attentive:
                assert projection_attention_f is not None

            self.batch_size = batch_size = batch_size
            self.num_steps = num_steps
            self.num_steps_valid = num_steps_valid
            vocab_size = vocab_size

            self._input_data_train = []
            self._targets_train = []
            self.mask_train = []

            for i in xrange(num_steps):  # Last bucket is the biggest one.
                self.input_data_train.append(tf.placeholder(tf.int32, shape=[None], name="input_train{0}".format(i)))
                self.targets_train.append(tf.placeholder(tf.int32, shape=[None], name="target_train{0}".format(i)))
                self.mask_train.append(tf.placeholder(tf.float32, shape=[None], name="mask_train{0}".format(i)))

            self._input_data_valid = []
            self._targets_valid = []
            self.mask_valid = []

            for i in xrange(num_steps_valid):  # Last bucket is the biggest one.
                self.input_data_valid.append(tf.placeholder(tf.int32, shape=[None], name="input_valid{0}".format(i)))
                self.targets_valid.append(tf.placeholder(tf.int32, shape=[None], name="target_valid{0}".format(i)))
                self.mask_valid.append(tf.placeholder(tf.float32, shape=[None], name="mask_valid{0}".format(i)))

            hidden_projection = None
            if hidden_proj > 0:
                hidden_projection = hidden_proj

            self.cell = cells.build_lm_multicell_rnn(num_layers, hidden_size, proj_size, use_lstm=use_lstm,
                                                     hidden_projection=hidden_projection, dropout=dropout_rate)

            self.dropout_feed = tf.placeholder(tf.float32, name="dropout_rate")

            self._initial_state_train = self.cell.zero_state(batch_size, tf.float32)
            self._initial_state_valid = self.cell.zero_state(1, tf.float32)

            # learning rate ops
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * lr_decay)

            # epoch ops
            self.epoch = tf.Variable(0, trainable=False)
            self.epoch_update_op = self.epoch.assign(self.epoch + 1)

            # samples seen ops
            self.samples_seen = tf.Variable(0, trainable=False)
            self.samples_seen_update_op = self.samples_seen.assign(self.samples_seen + batch_size)
            self.samples_seen_reset_op = self.samples_seen.assign(0)

            # global step variable - controled by the model
            self.global_step = tf.Variable(0.0, trainable=False)

            # average loss ops
            self.current_ppx = tf.Variable(1.0, trainable=False)
            self.current_loss = tf.Variable(0.0, trainable=False)
            # self.current_loss_update_op = None

            self.best_eval_ppx = tf.Variable(numpy.inf, trainable=False)
            self.estop_counter = tf.Variable(0, trainable=False)
            self.estop_counter_update_op = self.estop_counter.assign(self.estop_counter + 1)
            self.estop_counter_reset_op = self.estop_counter.assign(0)

            initializer = tf.random_uniform_initializer(minval=init_scale, maxval=init_scale, seed=_SEED)

            out_proj = hidden_size
            if hidden_proj > 0:
                out_proj = hidden_proj

            with tf.device("/cpu:0"):
                w = tf.get_variable("proj_w", [out_proj, vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [vocab_size])
            self.output_projection = (w, b)

            sampled_softmax = False

            # Sampled softmax only makes sense if we sample less than vocabulary size.
            if 0 < num_samples < vocab_size:
                sampled_softmax = True

                def sampled_loss(logits, labels):
                    with tf.device("/cpu:0"):
                        labels = tf.reshape(labels, [-1, 1])
                        losses = tf.nn.sampled_softmax_loss(w_t, b, logits, labels, num_samples, vocab_size)
                        return losses

                loss_function = sampled_loss

            with tf.device("/cpu:0"):
                # input come as one big tensor so we have to split it into a list of tensors to run the rnn cell
                embedding = tf.Variable(
                    tf.random_uniform(
                        [vocab_size, proj_size],
                        minval=-init_scale, maxval=init_scale
                    ),
                    name="embedding"
                )
                # embedding = tf.get_variable("embedding", [vocab_size, proj_size])

                inputs_train = [tf.nn.embedding_lookup(embedding, i) for i in self.input_data_train]
                inputs_valid = [tf.nn.embedding_lookup(embedding, i) for i in self.input_data_valid]

            with tf.variable_scope("RNN", initializer=initializer):

                if attentive:
                    outputs_train, state_train, _ = lm_ops.apply_attentive_lm(
                        self.cell, inputs_train, sequence_length=array_ops.squeeze(math_ops.add_n(self.mask_train)),
                        projection_attention_f=projection_attention_f, output_form=output_form,
                        dropout=self.dropout_feed, initializer=initializer, dtype=tf.float32
                    )

                    outputs_valid, state_valid, _ = lm_ops.apply_attentive_lm(
                        self.cell, inputs_valid, sequence_length=array_ops.squeeze(math_ops.add_n(self.mask_valid)),
                        projection_attention_f=projection_attention_f, output_form=output_form,
                        dropout=self.dropout_feed, initializer=initializer, dtype=tf.float32
                    )

                else:
                   outputs_train, state_train = lm_ops.apply_lm(
                       self.cell, inputs_train, sequence_length=math_ops.add_n(self.mask_train),
                       dropout=self.dropout_feed, dtype=tf.float32
                   )

                   outputs_valid, state_valid = lm_ops.apply_lm(
                       self.cell, inputs_valid, sequence_length=math_ops.add_n(self.mask_valid),
                       dropout=self.dropout_feed, dtype=tf.float32
                   )

                if sampled_softmax is False:
                    logits_train = [tf.nn.xw_plus_b(o, self.output_projection[0], self.output_projection[1])
                                    for o in outputs_train]
                    logits_valid = [tf.nn.xw_plus_b(o, self.output_projection[0], self.output_projection[1])
                                    for o in outputs_valid]
                else:
                    logits_train = outputs_train
                    logits_valid = outputs_valid

            loss_train = seq2seq.sequence_loss_by_example(
                logits_train, self.targets_train, self.mask_train, average_across_timesteps=True
            )

            loss_valid = seq2seq.sequence_loss_by_example(
                logits_valid, self.targets_valid, self.mask_valid, average_across_timesteps=True
            )

            self._cost_train = cost = tf.reduce_sum(loss_train) / float(batch_size)
            self._final_state_train = state_train

            self._cost_valid = tf.reduce_sum(loss_valid) / float(batch_size)
            self._final_state_valid = state_valid

            if not is_training:
                return

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                              max_grad_norm)

            opt = optimization_ops.get_optimizer(optimizer, learning_rate)
            self._train_op = opt.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            self._valid_op = tf.no_op()

            self.saver = tf.train.Saver(tf.all_variables())
            self.saver_best = tf.train.Saver(tf.all_variables())

    @property
    def input_data_train(self):
        return self._input_data_train

    @property
    def targets_train(self):
        return self._targets_train

    @property
    def initial_state_train(self):
        return self._initial_state_train

    @property
    def cost_train(self):
        return self._cost_train

    @property
    def final_state_train(self):
        return self._final_state_train

    @property
    def train_op(self):
        return self._train_op

    @property
    def input_data_valid(self):
        return self._input_data_valid

    @property
    def targets_valid(self):
        return self._targets_valid

    @property
    def initial_state_valid(self):
        return self._initial_state_valid

    @property
    def cost_valid(self):
        return self._cost_valid

    @property
    def final_state_valid(self):
        return self._final_state_valid

    @property
    def valid_op(self):
        return self._valid_op

    def train_step(self, session, lm_inputs, lm_targets, mask, state, dropout_rate=0.0):

        input_feed = {}
        for l in xrange(self.num_steps):
            input_feed[self.input_data_train[l].name] = lm_inputs[l]
            input_feed[self.targets_train[l].name] = lm_targets[l]
            input_feed[self.mask_train[l].name] = mask[l]

        input_feed[self.initial_state_train] = state
        input_feed[self.dropout_feed] = dropout_rate

        output_feed = [self.cost_train, self.final_state_train, self.train_op]

        cost_, state_, _ = session.run(output_feed, feed_dict=input_feed)

        return cost_, state_

    def valid_step(self, session, lm_inputs, lm_targets, mask, state, dropout_rate=0.0):

        input_feed = {}
        for l in xrange(self.num_steps_valid):
            input_feed[self.input_data_valid[l].name] = lm_inputs[l]
            input_feed[self.targets_valid[l].name] = lm_targets[l]
            input_feed[self.mask_valid[l].name] = mask[l]

        input_feed[self.initial_state_valid] = state
        input_feed[self.dropout_feed] = dropout_rate

        output_feed = [self.cost_valid, self.final_state_valid, self.valid_op]

        cost_, state_, _ = session.run(output_feed, feed_dict=input_feed)

        return cost_, state_