from __future__ import print_function
import tensorflow as tf
from train_ops import train_lm


flags = tf.flags
logging = tf.logging

tf.app.flags.DEFINE_string('model_name', 'lm_gru_1lr_256_proj256_en30000_maxNrm5_adam_dropout-off_input-feed-off.ckpt', 'Model name')
tf.app.flags.DEFINE_string('train_dir', '/home/gian/train_lms/lm_gru_1lr_256_proj256_en30000_maxNrm5_adam_dropout-off_input-feed-off/', 'Train directory')
tf.app.flags.DEFINE_string('best_models_dir', '/home/gian/train_lms/', 'Train directory')
tf.app.flags.DEFINE_string('data_dir', '/home/gian/data/', 'Data directory')
tf.app.flags.DEFINE_string('train_data', 'fapesp-v2.pt-en.train.tok.%s', 'Data for training.')
tf.app.flags.DEFINE_string('valid_data', 'fapesp-v2.pt-en.dev.tok.%s', 'Data for validation.')
tf.app.flags.DEFINE_string('test_data', 'fapesp-v2.pt-en.test-a.tok.%s', 'Data for testing.')
tf.app.flags.DEFINE_string('source_lang', 'en', 'Source language extension.')
tf.app.flags.DEFINE_integer('max_train_data_size', 0, 'Limit on the size of training data (0: no limit).')

flags.DEFINE_integer("src_vocab_size", 30000, "Size of the vocabulary to be used.")

tf.app.flags.DEFINE_string('optimizer', 'adam', 'Name of the optimizer to use (adagrad, adam, rmsprop or sgd).')
flags.DEFINE_float("learning_rate", 0.001, "The learning rate used when updating the LM parameters.")
flags.DEFINE_integer('start_decay', 0, 'Start learning rate decay at this epoch. Set to 0 to use patience.')
flags.DEFINE_integer('stop_decay', 0, 'Stop learning rate decay at this epoch. Set to 0 to use patience.')
flags.DEFINE_float("lr_decay", 1.0, "Decay the learning rate by this much. If 1.0, no decaying will be applied.")

flags.DEFINE_integer('num_samples_loss', 0, 'Number of samples to use in sampled softmax. Set to 0 to use regular loss.')
flags.DEFINE_float("init_scale", 0.05, "The scale to use when initializing the LM weights and biases")
flags.DEFINE_integer("num_layers", 2, "Number of hidden layers to use within the LM.")
flags.DEFINE_boolean('use_lstm', False, 'Whether to use LSTM units. Default to False.')
flags.DEFINE_integer('proj_size', 256, 'Size of words projection.')
flags.DEFINE_integer("hidden_size", 256, "Number of hidden units to use within the hidden layers.")

flags.DEFINE_float("max_grad_norm", 5.0, "Maximum L2 norm of the gradients before clipping.")

flags.DEFINE_integer("max_epochs", 39, "Maximum nnumber of epochs to train the LM.")
flags.DEFINE_integer("batch_size", 32, "Mini-batch size.")
flags.DEFINE_integer("num_steps", 35, "Maximum number of steps to unroll the network.")
flags.DEFINE_float("dropout_rate", 0.0, "The dropout rate to be applied to the LM when training.")

# verbosity and checkpoints
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 500, 'How many training steps to do per checkpoint.')
tf.app.flags.DEFINE_integer('steps_per_validation', 1000, 'How many training steps to do between each validation.')
tf.app.flags.DEFINE_integer('steps_verbosity', 10, 'How many training steps to do between each information print.')

# pacience flags (learning_rate decay and early stop)
flags.DEFINE_integer('lr_rate_patience', 3, 'How many training steps to monitor.')
flags.DEFINE_integer('early_stop_patience', 20, 'How many training steps to monitor.')
flags.DEFINE_integer('early_stop_after_epoch', 20, 'Start monitoring early_stop after this epoch.')
flags.DEFINE_boolean('save_best_model', True, 'Set to True to save the best model even if not using early stop.')

FLAGS = flags.FLAGS


def main(unused_args):
    train_lm(FLAGS)


if __name__ == "__main__":
    tf.app.run()