# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops, math_ops, nn_ops
from tensorflow.python.ops import variable_scope as vs

import cells

TYPE_1 = 'type_1'
TYPE_2 = 'type_2'


def get_attentive_content_f(name):
    if name == TYPE_1:
        return type_1
    elif name == TYPE_2:
        return type_2
    else:
        return "None"


def type_1(decoder_hidden, attn_size, initializer=None):

    with vs.variable_scope("decoder_type_1", initializer=initializer):

        k = vs.get_variable("AttnDecW_%d" % 0, [1, 1, attn_size, 1], initializer=initializer)
        hidden_features = nn_ops.conv2d(decoder_hidden, k, [1, 1, 1, 1], "SAME")

        # s will be (?, timesteps)
        s = math_ops.reduce_sum(math_ops.tanh(hidden_features), [2, 3])

    return s


def type_2(decoder_hidden, attn_size, initializer=None):

    with vs.variable_scope("decoder_type_2", initializer=initializer):

        k = vs.get_variable("AttnDecW_%d" % 0, [1, 1, attn_size, attn_size], initializer=initializer)
        hidden_features = nn_ops.conv2d(decoder_hidden, k, [1, 1, 1, 1], "SAME")
        v = vs.get_variable("AttnDecV_%d" % 0, [attn_size])

        # s will be (?, timesteps)
        s = math_ops.reduce_sum((v * math_ops.tanh(hidden_features)), [2, 3])

    return s
