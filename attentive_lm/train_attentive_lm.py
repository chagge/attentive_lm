from __future__ import print_function
import numpy
import tensorflow as tf
from train_ops import train_lm_traverse, train_lm
import content_functions
import lm_ops


flags = tf.flags
logging = tf.logging

model_name = "att2_out-split_lstm_2lr_emb650_hid650_proj256_en10000_batch32_steps50_maxNrm5_adam_0-0001_decay0_1-0_dropout-0-2_input-feed-off-data-PTB"
dir = "/home/gian/"

flags.DEFINE_string('model_name', model_name + ".ckpt", 'Model name')
flags.DEFINE_string('train_dir', dir + 'train_lms/' + model_name + '/', 'Train directory')
flags.DEFINE_string('best_models_dir', dir + 'train_lms/', 'Train directory')
flags.DEFINE_string('data_dir', dir + 'data/', 'Data directory')
tf.app.flags.DEFINE_string('train_data', 'ptb.train.tok.%s', 'Data for training.')
tf.app.flags.DEFINE_string('valid_data', 'ptb.valid.tok.%s', 'Data for validation.')
tf.app.flags.DEFINE_string('test_data', 'ptb.test.tok.%s', 'Data for testing.')
flags.DEFINE_string('source_lang', 'en', 'Source language extension.')
flags.DEFINE_integer('max_train_data_size', 0, 'Limit on the size of training data (0: no limit).')

flags.DEFINE_integer("src_vocab_size", 10000, "Size of the vocabulary to be used.")

flags.DEFINE_boolean('traverse_optimization', False, 'Whether or not we should use traverse optimization or not.')
flags.DEFINE_string('optimizer', 'adam', 'Name of the optimizer to use (adadelta, adagrad, adam, rmsprop or sgd).')
flags.DEFINE_float("learning_rate", 0.0001, "The learning rate used when updating the LM parameters.")
flags.DEFINE_integer('start_decay', 0, 'Start learning rate decay at this epoch. Set to 0 to use patience.')
flags.DEFINE_integer('stop_decay', 0, 'Stop learning rate decay at this epoch. Set to 0 to use patience.')
flags.DEFINE_float("lr_decay", 1.0, "Decay the learning rate by this much. If 1.0, no decaying will be applied.")

flags.DEFINE_integer('num_samples_loss', 0, 'Number of samples to use in sampled softmax. Set to 0 to use regular loss.')
flags.DEFINE_float("init_scale", 0.05, "The scale to use when initializing the LM weights and biases")
flags.DEFINE_integer("num_layers", 2, "Number of hidden layers to use within the LM.")
flags.DEFINE_boolean('use_lstm', True, 'Whether to use LSTM units. Default to False.')
flags.DEFINE_integer('proj_size', 650, 'Size of words projection.')
flags.DEFINE_integer("hidden_size", 650, "Number of hidden units to use within the hidden layers.")
flags.DEFINE_integer('hidden_proj', 256, 'Size of hidden projection projection. Default to 0 (no projection).')

flags.DEFINE_boolean('attentive', True, 'Whether to pay attention on the outputs. Default to False.')
flags.DEFINE_string('projection_attention', content_functions.TYPE_2, 'Which attention function to apply to the outputs. Default to None.')
flags.DEFINE_string('output_form', lm_ops.OUTPUT_CONCAT, 'Which type of combination to use to produce predictions. Default to concat.')

flags.DEFINE_float("max_grad_norm", 5.0, "Maximum L2 norm of the gradients before clipping.")

flags.DEFINE_integer("max_epochs", 39, "Maximum nnumber of epochs to train the LM.")
flags.DEFINE_integer("batch_size", 32, "Mini-batch size.")
flags.DEFINE_boolean('reset_state', False, 'Whether to reset the network states between the minibatches. Default to False.')
flags.DEFINE_integer("num_steps", 50, "Maximum number of steps to unroll the network.")
flags.DEFINE_integer("num_valid_steps", 101, "Maximum number of steps to unroll the network.")
flags.DEFINE_float("dropout_rate", 0.2, "The dropout rate to be applied to the LM when training.")

# verbosity and checkpoints
flags.DEFINE_integer('steps_per_checkpoint', 0, 'How many training steps to do per checkpoint. Set to 0 to save only after each epoch.')
flags.DEFINE_integer('steps_per_validation', 0, 'How many training steps to do between each validation.')
flags.DEFINE_integer('steps_verbosity', 10, 'How many training steps to do between each information print.')

# pacience flags (learning_rate decay and early stop)
flags.DEFINE_integer('lr_rate_patience', 0, 'How many training steps to monitor.')
flags.DEFINE_integer('early_stop_patience', 10, 'How many training steps to monitor. Set to 0 to ignore.')
flags.DEFINE_integer('early_stop_after_epoch', 1, 'Start monitoring early_stop after this epoch.')
flags.DEFINE_boolean('eval_after_each_epoch', True, 'Run eval after each epoch on the valid set.')
flags.DEFINE_boolean('test_after_each_epoch', True, 'Run eval after each epoch on the test set.')
flags.DEFINE_boolean('save_each_epoch', True, 'Whether ot nor to save at the end of each epoch.')
flags.DEFINE_boolean('save_best_model', True, 'Set to True to save the best model even if not using early stop.')

FLAGS = flags.FLAGS


def main(unused_args):
    if FLAGS.traverse_optimization:
        train_lm_traverse(FLAGS)
    else:
        train_lm(FLAGS)


if __name__ == "__main__":
    tf.app.run()
