import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops, control_flow_ops, math_ops, nn_ops, rnn
from tensorflow.python.ops import variable_scope as vs

import cells

_SEED = 1234
OUTPUT_CONCAT = "concat"
OUTPUT_SPLIT = "split"
OUTPUT_SINGLE = "single"


def apply_attentive_lm(cell, inputs, sequence_length=None, projection_attention_f=None, initializer=None,
                       dropout=None, output_form=OUTPUT_CONCAT, dtype=tf.float32):
    """

    Parameters
    ----------
    cell
    inputs
    sequence_length
    projection_attention_f
    initializer
    dropout
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
    (fixed_batch_size, _) = inputs[0].get_shape().with_rank(2)
    output_size = cell.output_size
    attn_size = cell.output_size
    outputs = []

    cell_outputs, cell_state = rnn.rnn(cell=cell,
                                       inputs=inputs,
                                       sequence_length=sequence_length,
                                       dtype=dtype)

    if sequence_length is not None:
        sequence_length = math_ops.to_int32(sequence_length)

    if sequence_length is not None:  # Prepare variables
        zero_output = array_ops.zeros(array_ops.pack([batch_size, cell.output_size]), inputs[0].dtype)
        zero_output.set_shape(tensor_shape.TensorShape([fixed_batch_size.value, cell.output_size]))
        min_sequence_length = math_ops.reduce_min(sequence_length)
        max_sequence_length = math_ops.reduce_max(sequence_length)

    for time, cell_output in enumerate(cell_outputs):
        if time > 0:
            vs.get_variable_scope().reuse_variables()

        cell_outs = cell_outputs[0:time + 1]  # we add +1 to t to get up to the output at t

        shape1 = len(cell_outs)

        top_states = [tf.reshape(o, [-1, 1, attn_size]) for o in cell_outs]
        output_attention_states = tf.concat(1, top_states)

        decoder_hidden = array_ops.reshape(output_attention_states, [-1, shape1, 1, attn_size])

        output_f = lambda: _output_with_attention(cell_output, output_size, decoder_hidden, attn_size,
                                                  projection_attention_f, initializer=initializer,
                                                  output_form=output_form)

        if sequence_length is not None:
            output = _combine_output_and_attention_step(time, sequence_length, min_sequence_length,
                                                        max_sequence_length, zero_output, output_f)
        else:
            output = output_f()

        outputs.append(output)

    cell_outs = [tf.reshape(o, [-1, 1, 1, attn_size]) for o in cell_outputs]

    cell_outputs = tf.concat(1, cell_outs)

    return outputs, cell_state, cell_outputs


def _combine_output_and_attention_step(time, sequence_length, min_sequence_length, max_sequence_length,
                                       zero_output, output_f):

    def _copy_some_through(new_output):
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        copy_cond = (time >= sequence_length)
        return math_ops.select(copy_cond, zero_output, new_output)

    def _maybe_copy_some_through():
        """Run RNN step.  Pass through either no or some past state."""
        new_output = output_f()

        return control_flow_ops.cond(
            # if t < min_seq_len: calculate and return everything
            time < min_sequence_length, lambda: new_output,
            # else copy some of it through
            lambda: _copy_some_through(new_output))

    empty_update = lambda: zero_output

    final_output = control_flow_ops.cond(
        # if t >= max_seq_len: copy all state through, output zeros
        time >= max_sequence_length, empty_update,
        # otherwise calculation is required: copy some or all of it through
        _maybe_copy_some_through)

    final_output.set_shape(zero_output.get_shape())

    return final_output


def _output_with_attention(cell_output, output_size, decoder_hidden, attn_size,
                           projection_attention_f, initializer=None, output_form=OUTPUT_CONCAT):
    """

    Parameters
    ----------
    decoder_hidden
    attn_size
    projection_attention_f
    initializer
    step_num

    Returns
    -------

    """
    assert initializer is not None

    with vs.variable_scope("AttnOutputProjection", initializer=initializer):

        with vs.variable_scope("output_attention", initializer=initializer):

            s = projection_attention_f(decoder_hidden, attn_size)

            # beta will be (?, timesteps)
            beta = nn_ops.softmax(s)

            shape = decoder_hidden.get_shape()
            timesteps = shape[1].value
            b = array_ops.reshape(beta, [-1, timesteps, 1, 1])

            # b  and decoder_hidden will be (?, timesteps, 1, 1)
            d = math_ops.reduce_sum(b * decoder_hidden, [1, 2])

            # d is (?, decoder_size)
            # ds is (?, decoder_size)
            ds = tf.reshape(d, [-1, attn_size])

            _ = tf.histogram_summary('attention_context', ds)

        # output = cells.linear([cell_output] + [ds], output_size, True)

        if output_form == OUTPUT_SPLIT:
            output = _output_form_split(cell_output, ds, output_size, initializer=initializer)

        elif output_form == OUTPUT_SINGLE:
            output = _output_form_single(ds, output_size, initializer=initializer)

        else:
            output = _output_form_concat(cell_output, ds, output_size, initializer=initializer)

        # output = tf.tanh(output)
        output = nn_ops.relu(output)

    return output


def _output_form_concat(cell_output, decoder_states, output_size, initializer):

    with vs.variable_scope("output_form", initializer=initializer):
        output = cells.linear([cell_output] + [decoder_states], output_size, True)

    return output


def _output_form_single(decoder_states, output_size, initializer):

    with vs.variable_scope("decoder_states_linear", initializer=initializer):
        output = cells.linear([decoder_states], output_size, True)

    return output


def _output_form_split(cell_output, decoder_states, output_size, initializer):

    with vs.variable_scope("cell_output_linear", initializer=initializer):
        cell_output_ = cells.linear([cell_output], output_size, True)

    with vs.variable_scope("decoder_states_linear", initializer=initializer):
        ds_ = cells.linear([decoder_states], output_size, True)

    output = cell_output_ + ds_

    return output


def apply_lm(cell, inputs, sequence_length=None, dropout=None, dtype=tf.float32):
    """

    Parameters
    ----------
    cell
    inputs
    sequence_length
    dropout
    dtype

    Returns
    -------

    """
    if dropout is not None:

        for c in cell._cells:
            c.input_keep_prob = 1.0 - dropout

    cell_outputs, cell_state = rnn.rnn(cell=cell,
                                       inputs=inputs,
                                       sequence_length=sequence_length,
                                       dtype=dtype)

    return cell_outputs, cell_state
