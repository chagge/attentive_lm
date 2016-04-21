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
# from six.moves import xrange


def train_lm(FLAGS=None):

    assert FLAGS is not None

    print('Preparing data in %s' % FLAGS.data_dir)
    src_train, src_dev, src_test = prepare_lm_data(FLAGS)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

        nan_detected = False

        print('Creating layers.')
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        model = build_ops.create_lm_model(sess, is_training=True, FLAGS=FLAGS, initializer=initializer)

        print('Reading development and training data (limit: %d).' % FLAGS.max_train_data_size)

        valid_data = read_lm_data(src_dev, FLAGS=FLAGS)
        test_data = read_lm_data(src_test, FLAGS=FLAGS)
        train_data = read_lm_data(src_train, max_size=FLAGS.max_train_data_size, FLAGS=FLAGS)
        train_total_size = len(train_data)

        epoch_size = train_total_size / FLAGS.batch_size
        print("Total number of updates per epoch: %d" % epoch_size)

        step_time = 0.0
        words_time = 0.0
        n_target_words = 0

        print("Optimization started...")
        while model.epoch.eval() < FLAGS.max_epochs:

            saved = False

            start_time = time.time()

            lm_inputs, lm_targets, lm_mask, n_words = model.get_train_batch(train_data)

            n_target_words += n_words

            loss, _ = model.train_step(session=sess, lm_inputs=lm_inputs, lm_targets=lm_targets, mask=lm_mask)

            currloss = model.current_loss.eval()
            sess.run(model.current_loss.assign(currloss + loss))
            sess.run(model.samples_seen_update_op)

            current_step = model.global_step.eval()

            if numpy.isnan(loss) or numpy.isinf(loss):
                print 'NaN detected'
                nan_detected = True
                break

            if current_step % FLAGS.steps_verbosity == 0:

                closs = model.current_loss.eval()
                gstep = model.global_step.eval()
                avgloss = closs / gstep
                sess.run(model.avg_loss.assign(avgloss))

                if words_time == 0.0:
                    words_time += (time.time() - start_time)

                target_words_speed = n_target_words / words_time

                loss = model.avg_loss.eval()
                ppx = math.exp(loss) if loss < 300 else float('inf')

                if ppx > 1000.0:

                    print(
                        'epoch %d gl.step %d lr.rate %.4f steps-time %.2f avg.loss %.8f avg.ppx > 1000.0 - avg. %.2f K target words/sec' %
                        (model.epoch.eval(), model.global_step.eval(), model.learning_rate.eval(),
                         step_time, loss, (target_words_speed / 1000.0)))

                else:

                    print(
                        'epoch %d gl.step %d lr.rate %.4f steps-time %.2f avg.loss %.8f avg.ppx %.8f - avg. %.2f K target words/sec' %
                        (model.epoch.eval(), model.global_step.eval(), model.learning_rate.eval(),
                         step_time, loss, ppx, (target_words_speed / 1000.0)))

                n_target_words = 0
                step_time = 0.0
                words_time = 0.0

            if FLAGS.steps_per_checkpoint > 0:
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    # Save checkpoint
                    checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    saved = True

            # update epoch number
            if model.samples_seen.eval() >= train_total_size:
                sess.run(model.epoch_update_op)
                ep = model.epoch.eval()
                print("Epoch %d finished..." % (ep - 1))

                should_stop = False

                if FLAGS.eval_after_each_epoch:

                    print("\nValidating:\n")

                    avg_eval_loss, avg_ppx = run_eval(model=model, session=sess, data=valid_data,
                                                      batch_size=FLAGS.batch_size)

                    best_loss = model.best_eval_loss.eval()
                    best_ppx = math.exp(best_loss) if best_loss < 300 else float('inf')

                    with codecs.open(FLAGS.best_models_dir + FLAGS.model_name + ".txt", "a", encoding="utf-8") as f:
                        f.write("PPX after epoch #%d: %f (Loss: %f  - #steps: %d - Current best PPX: %f)\n" %
                                (ep, avg_ppx, avg_eval_loss, current_step, best_ppx))


                    print("PPX after epoch #%d: %f (Loss: %f  - #steps: %d - Current best PPX: %f)" %
                            (ep, avg_ppx, avg_eval_loss, current_step, best_ppx))

                    if FLAGS.steps_per_validation == 0:

                        # if we are not validating after some steps, we validate after each epoch,
                        # therefore we must check the early stop here
                        should_stop = check_early_stop(model=model, session=sess, loss=avg_eval_loss, flags=FLAGS)

                # Save checkpoint
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                if should_stop:
                    break

                print("Epoch %d started..." % ep)
                sess.run(model.samples_seen_reset_op)

                if ep >= FLAGS.max_epochs:
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

            if FLAGS.steps_per_validation > 0:

                if current_step % FLAGS.steps_per_validation == 0:

                    print("\nValidating:\n")

                    avg_eval_loss, avg_ppx = run_eval(model=model, session=sess, data=valid_data,
                                                      batch_size=FLAGS.batch_size)

                    if avg_ppx > 1000.0:
                        print('\n  eval: averaged perplexity > 1000.0')
                    else:
                        print('\n  eval: averaged perplexity %.8f' % avg_ppx)
                    print('  eval: averaged loss %.8f\n' % avg_eval_loss)

                    should_stop = check_early_stop(model=model, session=sess, loss=avg_eval_loss, flags=FLAGS)

                    if should_stop:
                        break

            step_time += (time.time() - start_time) / FLAGS.steps_verbosity
            words_time += (time.time() - start_time)

        print("\nTraining finished!!\n")

        if not nan_detected:
            # # Save checkpoint
            checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)

            print("Final validation:")

            avg_eval_loss, avg_ppx = run_eval(model=model, session=sess, data=valid_data, batch_size=FLAGS.batch_size)

            if avg_ppx > 1000.0:
                print('\n  eval: averaged valid. perplexity > 1000.0')
            else:
                print('\n  eval: averaged valid. perplexity %.8f' % avg_ppx)
            print('  eval: averaged valid. loss %.8f\n' % avg_eval_loss)

            print("\n##### Test Results: #####\n")

            avg_test_loss, test_ppx = run_eval(model=model, session=sess, data=test_data, batch_size=FLAGS.batch_size)

            if test_ppx > 1000.0:
                print('\n  eval: averaged test perplexity > 1000.0')
            else:
                print('\n  eval: averaged test perplexity %.8f' % test_ppx)
            print('  eval: averaged test loss %.8f\n' % avg_test_loss)

            sys.stdout.flush()


def run_eval(model, session, data, batch_size):
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
    eval_batch_size = len(data)
    eval_steps = eval_batch_size / batch_size
    last_step = eval_batch_size % batch_size

    avg_eval_loss = 0.0

    for i in xrange(eval_steps + 1):

        b = batch_size
        if i == eval_steps:
            d = data[-last_step:]
            eval_inputs, eval_targets, eval_mask, _ = model.get_train_batch(d, batch=last_step)
        else:
            d = data[(i * b):((i + 1) * b)]
            eval_inputs, eval_targets, eval_mask, _ = model.get_train_batch(d, batch=b)

        eval_cost, _ = model.train_step(session=session, lm_inputs=eval_inputs,
                                        lm_targets=eval_targets,
                                        mask=eval_mask, op=tf.no_op())

        avg_eval_loss += eval_cost

    avg_eval_loss = avg_eval_loss / (eval_steps + 1)

    eval_ppx = math.exp(avg_eval_loss) if avg_eval_loss < 300 else float('inf')

    return avg_eval_loss, eval_ppx


def check_early_stop(model, session, loss, flags):
    """

    Parameters
    ----------
    model
    session
    loss
    flags

    Returns
    -------

    """

    stop = False

    patience = flags.early_stop_patience

    # check early stop - if early stop patience is greater than 0, test it
    if patience > 0:

        if loss < model.best_eval_loss.eval():
            session.run(model.best_eval_loss.assign(loss))
            session.run(model.estop_counter_reset_op)
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

        print('\n   best valid. loss: %.8f' % model.best_eval_loss.eval())
        print('early stop patience: %d - max %d\n' % (int(model.estop_counter.eval()), patience))

    return stop