# -*- coding: utf-8 -*-
import codecs
import math
import numpy
import os
import tensorflow as tf
import time
import sys
from tensorflow.python.platform import gfile
import build_ops
from data_utils import read_lm_data, prepare_lm_data
import reader
# from six.moves import xrange


def train_lm(FLAGS=None):

    assert FLAGS is not None

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

    if not os.path.exists(FLAGS.best_models_dir):
        os.makedirs(FLAGS.best_models_dir)

    src_lang = FLAGS.source_lang
    # raw_data = reader.ptb_raw_data(FLAGS.data_dir,
    #                                train=FLAGS.train_data % src_lang,
    #                                valid=FLAGS.train_data % src_lang,
    #                                test=FLAGS.train_data % src_lang,
    #                                vocab_size=FLAGS.src_vocab_size)
    #
    # train_data, valid_data, test_data, _ = raw_data

    print('Reading development and training data (limit: %d).' % FLAGS.max_train_data_size)

    src_train, src_dev, src_test = prepare_lm_data(FLAGS)

    valid_data = read_lm_data(src_dev, FLAGS=FLAGS)
    test_data = read_lm_data(src_test, FLAGS=FLAGS)
    train_data = read_lm_data(src_train, max_size=FLAGS.max_train_data_size, FLAGS=FLAGS)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

        nan_detected = False

        print('Creating layers.')
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        model = build_ops.create_lm_model(sess, is_training=True, FLAGS=FLAGS, initializer=initializer)

        print("Optimization started...")
        while model.epoch.eval() < FLAGS.max_epochs:

            saved = False

            start_time = time.time()

            epoch_size = ((len(train_data) / model.batch_size) - 1) / model.num_steps
            costs = 0.0
            iters = 0
            state = model.initial_state_train.eval()

            for step, (x, y) in enumerate(reader.ptb_iterator(train_data, model.batch_size,  model.num_steps)):
                loss, state, _ = sess.run([model.cost_train, model.final_state_train, model.train_op],
                                          {model.input_data_train: x, model.targets_train: y,
                                           model.initial_state_train: state,
                                           model.dropout_feed: FLAGS.dropout_rate})

                if numpy.isnan(loss) or numpy.isinf(loss):
                    print 'NaN detected'
                    nan_detected = True
                    break

                costs += loss
                iters += model.num_steps

                current_global_step = model.global_step.eval()

                if current_global_step % FLAGS.steps_verbosity == 0:

                    target_words_speed = (iters * model.batch_size) / (time.time() - start_time)
                    avg_step_time = iters / (time.time() - start_time)

                    loss = costs / iters
                    ppx = numpy.exp(loss)
                    sess.run(model.current_ppx.assign(ppx))
                    sess.run(model.current_loss.assign(ppx))

                    print('epoch %d global step %d lr.rate %.4f avg.loss %.4f avg.ppx %.8f step time %.2f - avg. %.2f words/sec' %
                          (model.epoch.eval(), current_global_step, model.learning_rate.eval(),
                           loss, ppx, avg_step_time, target_words_speed))

                if FLAGS.steps_per_checkpoint > 0:
                    if current_global_step % FLAGS.steps_per_checkpoint == 0:
                        # Save checkpoint
                        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                        saved = True

                if FLAGS.steps_per_validation > 0:

                    if current_global_step % FLAGS.steps_per_validation == 0:

                        print("\nValidating:\n")

                        valid_ppx = run_eval(model=model, session=sess, data=valid_data)
                        best_ppx = model.best_eval_ppx.eval()

                        print("PPX %f Global Step %d (Current best PPX: %f)\n" % (valid_ppx, step, best_ppx))

                        should_stop = check_early_stop(model=model, session=sess, ppx=valid_ppx, flags=FLAGS)

                        if should_stop:
                            break

            ep = model.epoch.eval()
            print("Epoch %d finished... " % ep)

            epoch_ppx = numpy.exp(costs / iters)

            should_stop = False
            ep_new = ep

            if FLAGS.save_each_epoch:
                # Save checkpoint
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                # updating epoch number
                sess.run(model.epoch_update_op)
                ep_new = model.epoch.eval()

            if FLAGS.eval_after_each_epoch:

                print("\nValidating:\n")

                valid_ppx = run_eval(model=model, session=sess, data=valid_data)
                best_ppx = model.best_eval_ppx.eval()

                with codecs.open(FLAGS.best_models_dir + FLAGS.model_name + ".txt", "a", encoding="utf-8") as f:
                    f.write("PPX after epoch #%d: %f (Current best PPX: %f)\n" % (ep, valid_ppx, best_ppx))

                print("Validation PPX after epoch #%d: %f (Current best PPX: %f)\n" % (ep, valid_ppx, best_ppx))

                if FLAGS.steps_per_validation == 0:
                    # if we are not validating after some steps, we validate after each epoch,
                    # therefore we must check the early stop here
                    should_stop = check_early_stop(model=model, session=sess, ppx=valid_ppx, flags=FLAGS)

            if ep + 1 >= FLAGS.max_epochs:
                if not saved:
                    # Save checkpoint
                    checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                break

            if FLAGS.start_decay > 0:

                if FLAGS.stop_decay > 0:

                    if FLAGS.start_decay <= model.epoch.eval() <= FLAGS.stop_decay:
                        sess.run(model.learning_rate_decay_op)

                else:

                    if FLAGS.start_decay <= model.epoch.eval():
                        sess.run(model.learning_rate_decay_op)

            if should_stop:
                break

            print("Epoch %d started..." % ep_new)
            sess.run(model.samples_seen_reset_op)

        # when we ran the right number of epochs or we reached early stop we finish training
        print("\nTraining finished!!\n")

        if not nan_detected:
            # # Save checkpoint
            checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)

            print("Final validation:")

            avg_eval_loss, avg_ppx = run_eval(model=model, session=sess, data=valid_data)

            if avg_ppx > 1000.0:
                print('\n  eval: averaged valid. perplexity > 1000.0')
            else:
                print('\n  eval: averaged valid. perplexity %.8f' % avg_ppx)
            print('  eval: averaged valid. loss %.8f\n' % avg_eval_loss)

            print("\n##### Test Results: #####\n")

            avg_test_loss, test_ppx = run_eval(model=model, session=sess, data=test_data)

            if test_ppx > 1000.0:
                print('\n  eval: averaged test perplexity > 1000.0')
            else:
                print('\n  eval: averaged test perplexity %.8f' % test_ppx)
            print('  eval: averaged test loss %.8f\n' % avg_test_loss)

            sys.stdout.flush()


def run_eval(model, session, data, batch_size=1, num_steps=1):
    """

    Parameters
    ----------
    model
    session
    data
    batch_size

    Returns
    -------

    """
    costs = 0.0
    iters = 0
    state = model.initial_state_train.eval()
    for step, (x, y) in enumerate(reader.ptb_iterator(data, batch_size, num_steps)):

        cost, state, _ = session.run([model.cost_valid, model.final_state_valid, model.valid_op],
                                     {model.input_data_valid: x,
                                      model.targets_valid: y,
                                      model.initial_state_valid: state,
                                      model.dropout_feed: 0.0})
        costs += cost
        iters += model.num_steps

    return numpy.exp(costs / iters)


def check_early_stop(model, session, ppx, flags):
    """

    Parameters
    ----------
    model
    session
    ppx
    flags

    Returns
    -------

    """

    stop = False

    patience = flags.early_stop_patience

    # check early stop - if early stop patience is greater than 0, test it
    if patience > 0:

        if ppx < model.best_eval_ppx.eval():
            session.run(model.best_eval_ppx.assign(ppx))
            session.run(model.estop_counter_reset_op)
            if flags.save_best_model:
                # Save checkpoint
                print('\nSaving the best model so far...')
                best_model_path = os.path.join(flags.best_models_dir, flags.model_name + '-best')
                model.saver_best.save(session, best_model_path, global_step=model.global_step)

        else:
            # if FLAGS.early_stop_after_epoch is equal to 0, it will monitor from the beginning
            if model.epoch.eval() >= flags.early_stop_after_epoch:

                session.run(model.estop_counter_update_op)

                if model.estop_counter.eval() >= patience:
                    print('\nEARLY STOP!\n')
                    stop = True

        print('\n   best valid. ppx: %.8f' % model.best_eval_ppx.eval())
        print('early stop patience: %d - max %d\n' % (int(model.estop_counter.eval()), patience))

    return stop