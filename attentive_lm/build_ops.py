import tensorflow as tf
from tensorflow.python.platform import gfile

import content_functions
import lm_models


def create_lm_model(session, is_training=True, FLAGS=None, initializer=None, model_path=None):

    assert FLAGS is not None
    assert initializer is not None

    with tf.variable_scope("model", reuse=None, initializer=initializer):

        if FLAGS.attentive:
            projection_attention_f = content_functions.get_attentive_content_f(FLAGS.projection_attention)
        else:
            projection_attention_f = None

        if is_training:

            model = lm_models.LMModel(is_training=is_training,
                                      learning_rate=FLAGS.learning_rate,
                                      max_grad_norm=FLAGS.max_grad_norm,
                                      num_layers=FLAGS.num_layers,
                                      num_steps=FLAGS.num_steps,
                                      proj_size=FLAGS.proj_size,
                                      hidden_size=FLAGS.hidden_size,
                                      use_lstm=FLAGS.use_lstm,
                                      num_samples=FLAGS.num_samples_loss,
                                      early_stop_patience=FLAGS.early_stop_patience,
                                      dropout_rate=FLAGS.dropout_rate,
                                      lr_decay=FLAGS.lr_decay,
                                      batch_size=FLAGS.batch_size,
                                      vocab_size=FLAGS.src_vocab_size,
                                      attentive=FLAGS.attentive,
                                      projection_attention_f=projection_attention_f)
        else:
            model = lm_models.LMModel(is_training=is_training,
                                      learning_rate=FLAGS.learning_rate,
                                      max_grad_norm=FLAGS.max_grad_norm,
                                      num_layers=FLAGS.num_layers,
                                      num_steps=1,
                                      proj_size=FLAGS.proj_size,
                                      hidden_size=FLAGS.hidden_size,
                                      use_lstm=FLAGS.use_lstm,
                                      num_samples=FLAGS.num_samples_loss,
                                      early_stop_patience=FLAGS.early_stop_patience,
                                      dropout_rate=FLAGS.dropout_rate,
                                      lr_decay=FLAGS.lr_decay,
                                      batch_size=1,
                                      vocab_size=FLAGS.src_vocab_size,
                                      attentive=FLAGS.attentive,
                                      projection_attention_f=projection_attention_f)

        if model_path is None:

            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

            if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
                print('Reading model parameters from %s' % ckpt.model_checkpoint_path)
                model.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print('Created model with fresh parameters.')
                session.run(tf.initialize_all_variables())

        else:
            print('Reading model parameters from %s' % model_path)
            model.saver.restore(session, model_path)

    return model

