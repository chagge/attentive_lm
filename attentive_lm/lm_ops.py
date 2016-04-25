import tensorflow as tf
from tensorflow.python.ops import array_ops, embedding_ops, math_ops, nn_ops
from tensorflow.python.ops import variable_scope as vs

import cells

_SEED = 1234


def apply_attentive_lm(cell, inputs, projection_attention_f=None, initializer=None,
                       dropout=None, dtype=tf.float32):
    """

    Parameters
    ----------
    cell
    inputs
    dtype

    Returns
    -------

    """

    assert projection_attention_f is not None

    if dropout is not None:

        for c in cell._cells:
            c.input_keep_prob = 1.0 - dropout

    if initializer is None:
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=_SEED)

    batch_size = array_ops.shape(inputs[0])[0]
    output_size = cell.output_size
    attn_size = cell.output_size

    cell_outputs = []
    outputs = []
    cell_state = cell.zero_state(batch_size=batch_size, dtype=dtype)

    # ct, _ = tf.split(split_dim=1, num_split=2, value=cell_state)
    #
    # cell_outputs.append(ct)

    for i in xrange(len(inputs)):
        if i > 0:
            vs.get_variable_scope().reuse_variables()

        cell_output, new_state = cell(inputs[i], cell_state)
        cell_state = new_state
        cell_outputs.append(cell_output)

        shape1 = len(cell_outputs)

        top_states = [tf.reshape(o, [-1, 1, attn_size]) for o in cell_outputs]

        output_attention_states = tf.concat(1, top_states)

        decoder_hidden = array_ops.reshape(output_attention_states, [-1, shape1, 1, attn_size])

        with vs.variable_scope("AttnOutputProjection", initializer=initializer):

            ht_hat = decoder_output_attention(decoder_hidden,
                                              attn_size,
                                              projection_attention_f,
                                              initializer=initializer)

            # with vs.variable_scope("AttnOutputProjection_logit_lstm", initializer=initializer):
            #     # if we pass a list of tensors, linear will first concatenate them over axis 1
            #     logit_lstm = cells.linear([ht_hat], output_size, True)
            #
            # with vs.variable_scope("AttnOutputProjection_logit_ctx", initializer=initializer):
            #     # if we pass a list of tensors, linear will first concatenate them over axis 1
            #     logit_ctx = cells.linear([cell_output], output_size, True)

            output = cells.linear([cell_output] + [ht_hat], output_size, True)

            output = tf.tanh(output)

            outputs.append(output)

    cell_outs = [tf.reshape(o, [-1, 1, 1, attn_size]) for o in cell_outputs]

    cell_outputs = tf.concat(1, cell_outs)

    return outputs, cell_state, cell_outputs


def decoder_output_attention(decoder_hidden, attn_size, decoder_attention_f, initializer=None, step_num=None):
    """

    Parameters
    ----------
    decoder_states
    attn_size

    Returns
    -------

    """
    assert initializer is not None

    with vs.variable_scope("output_attention", initializer=initializer):

        s = decoder_attention_f(decoder_hidden, attn_size)

        # beta will be (?, timesteps)
        beta = nn_ops.softmax(s)

        if step_num is None:  # step_num is None when training

            shape = decoder_hidden.get_shape()
            timesteps = shape[1].value
            b = array_ops.reshape(beta, [-1, timesteps, 1, 1])

        else:

            b = array_ops.reshape(beta, tf.pack([-1, step_num, 1, 1]))

        # b  and decoder_hidden will be (?, timesteps, 1, 1)
        d = math_ops.reduce_sum(b * decoder_hidden, [1, 2])

        # d will be (?, decoder_size)
        ds = tf.reshape(d, [-1, attn_size])

    _ = tf.histogram_summary('attention_context', ds)

    # ds is (?, decoder_size)
    return ds


def apply_lm(cell, inputs, dropout=None, dtype=tf.float32):
    """

    Parameters
    ----------
    cell
    inputs
    dtype

    Returns
    -------

    """
    if dropout is not None:

        for c in cell._cells:
            c.input_keep_prob = 1.0 - dropout

    cell_outputs = []
    cell_state = None

    batch_size = array_ops.shape(inputs[0])[0]
    cell_state = cell.zero_state(batch_size=batch_size, dtype=dtype)

    for i in xrange(len(inputs)):
        if i > 0:
            vs.get_variable_scope().reuse_variables()

        cell_output, new_state = cell(inputs[i], cell_state)

        cell_outputs.append(cell_output)
        cell_state = new_state

    return cell_outputs, cell_state