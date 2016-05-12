# -*- coding: utf-8 -*-
import codecs
import math
import numpy
import os
import tensorflow as tf
import time
import sys
import build_ops
import data_utils
import reader
# from six.moves import xrange


def train_lm_traverse(FLAGS=None):

    assert FLAGS is not None

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

    if not os.path.exists(FLAGS.best_models_dir):
        os.makedirs(FLAGS.best_models_dir)

    print('Reading development and training data (limit: %d).' % FLAGS.max_train_data_size)

    raw_data = reader.ptb_raw_data(FLAGS.data_dir, FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data)
    train_data, valid_data, test_data, _ = raw_data

    saved = True

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

        nan_detected = False

        print('Creating layers.')
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        model = build_ops.create_lm_model(sess, is_training=True, FLAGS=FLAGS, initializer=initializer)

        print("Optimization started...")
        while model.epoch.eval() < FLAGS.max_epochs:

            saved = False

            start_time = time.time()
            costs = 0.0
            iters = 0
            state = model.initial_state_train.eval()

            for step, (x, y) in enumerate(reader.ptb_iterator(train_data, model.batch_size,  model.num_steps)):
                loss, ppx, state, _ = sess.run([model.cost_train, model.ppx_train, model.final_state_train, model.train_op],
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
                    current_ppx = model.current_ppx.eval()
                    sess.run(model.current_ppx.assign(current_ppx * ppx))
                    sess.run(model.current_loss.assign(ppx))

                    print('epoch %d global step %d lr.rate %.4f avg.loss %.4f aprox.ppx %.8f step time %.2f - avg. %.2f words/sec' %
                          (model.epoch.eval(), current_global_step, model.learning_rate.eval(),
                           loss, ppx, avg_step_time, target_words_speed))

                if FLAGS.steps_per_checkpoint > 0:
                    if current_global_step % FLAGS.steps_per_checkpoint == 0:
                        # Save checkpoint
                        print("Saving current model...")
                        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                        saved = True

                if FLAGS.steps_per_validation > 0:

                    if current_global_step % FLAGS.steps_per_validation == 0:

                        print("\nValidating:\n")

                        valid_loss, valid_ppx = run_eval_traversing(model=model, session=sess, data=valid_data)
                        best_ppx = model.best_eval_ppx.eval()

                        print("PPX %f Global Step %d (Current best PPX: %f)\n" % (valid_ppx, step, best_ppx))

                        should_stop = check_early_stop(model=model, session=sess, ppx=valid_ppx, flags=FLAGS)

                        if should_stop:
                            break

            ep = model.epoch.eval()
            print("Epoch %d finished... " % ep)

            should_stop = False
            ep_new = ep

            if FLAGS.save_each_epoch:
                # updating epoch number
                sess.run(model.epoch_update_op)
                ep_new = model.epoch.eval()

                # Save checkpoint
                print("Saving current model...")
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

            if FLAGS.eval_after_each_epoch:

                print("\nValidating:\n")

                valid_loss, valid_ppx = run_eval_traversing(model=model, session=sess, data=valid_data)
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
            if not saved:
                # # Save checkpoint
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

            print("Final validation:")

            avg_eval_loss, avg_ppx = run_eval_traversing(model=model, session=sess, data=valid_data)

            if avg_ppx > 1000.0:
                print('\n  eval: averaged valid. perplexity > 1000.0')
            else:
                print('\n  eval: averaged valid. perplexity %.8f' % avg_ppx)
            print('  eval: averaged valid. loss %.8f\n' % avg_eval_loss)

            print("\n##### Test Results: #####\n")

            avg_test_loss, test_ppx = run_eval_traversing(model=model, session=sess, data=test_data)

            if test_ppx > 1000.0:
                print('\n  eval: averaged test perplexity > 1000.0')
            else:
                print('\n  eval: averaged test perplexity %.8f' % test_ppx)
            print('  eval: averaged test loss %.8f\n' % avg_test_loss)

            sys.stdout.flush()


def run_eval_traversing(model, session, data, batch_size=1, num_steps=120):
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
    eval_ppx = 0.0
    iters = 0
    state = model.initial_state_train.eval()
    for step, (x, y) in enumerate(reader.ptb_iterator(data, batch_size, num_steps)):

        cost, ppx, state, _ = session.run([model.cost_valid, model.ppx_valid, model.final_state_valid, model.valid_op],
                                     {model.input_data_valid: x,
                                      model.targets_valid: y,
                                      model.initial_state_valid: state,
                                      model.dropout_feed: 0.0})
        costs += cost
        iters += model.num_steps
        eval_ppx *= ppx
    eval_cost = costs / iters

    return eval_cost, eval_ppx


def train_lm(FLAGS=None):

    assert FLAGS is not None

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

    if not os.path.exists(FLAGS.best_models_dir):
        os.makedirs(FLAGS.best_models_dir)

    print('Preparing data in %s' % FLAGS.data_dir)
    src_train, src_dev, src_test = data_utils.prepare_lm_data(FLAGS)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

        nan_detected = False

        print('Creating layers.')
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        model = build_ops.create_lm_model(sess, is_training=True, FLAGS=FLAGS, initializer=initializer)

        print('Reading development and training data (limit: %d).' % FLAGS.max_train_data_size)

        train_data = data_utils.read_lm_data(src_train, max_size=FLAGS.max_train_data_size, FLAGS=FLAGS)
        train_total_size = len(train_data)
        m = max([len(s) for s in train_data])
        a = float(sum([len(s) for s in train_data])) / len(train_data)
        print("Train max length : %d - (Avg. %.2f)" % (m, a))

        valid_data = data_utils.read_lm_data(src_dev, FLAGS=FLAGS)
        m = max([len(s) for s in valid_data])
        print("Valid max length : %d" % m)

        test_data = data_utils.read_lm_data(src_test, FLAGS=FLAGS)
        m = max([len(s) for s in test_data])
        print("Test max length : %d" % m)

        epoch_size = train_total_size / FLAGS.batch_size
        print("Total number of updates per epoch: %d" % epoch_size)

        print("Optimization started...")

        total_loss = model.current_loss.eval()
        while model.epoch.eval() < FLAGS.max_epochs:

            saved = False
            n_target_words = 0
            state_ = model.initial_state_train.eval()

            for step, (x, y, w, words) in enumerate(data_utils.data_iterator(train_data, model.batch_size, model.num_steps)):

                start_time = time.time()

                if FLAGS.reset_state:
                    state = model.initial_state_train.eval()
                else:
                    state = state_

                n_target_words += words
                loss, state_ = model.train_step(session=sess, lm_inputs=x, lm_targets=y, mask=w,
                                                state=state, dropout_rate=FLAGS.dropout_rate)

                if numpy.isnan(loss) or numpy.isinf(loss):
                    print 'NaN detected'
                    nan_detected = True
                    break

                total_loss += loss
                current_global_step = model.global_step.eval()

                if current_global_step % FLAGS.steps_verbosity == 0:
                    end_time = time.time()
                    total_time = end_time - start_time
                    target_words_speed = n_target_words / total_time
                    n_target_words = 0
                    avg_step_time = total_time / FLAGS.steps_verbosity

                    avg_loss = total_loss / current_global_step
                    ppx = numpy.exp(avg_loss)
                    sess.run(model.current_loss.assign(total_loss))
                    sess.run(model.current_ppx.assign(ppx))

                    if ppx > 1000.0:

                        print('epoch %d global step %d lr.rate %.4f avg.loss %.4f avg. ppx > 1000.0 avg. step time %.2f - avg. %.2f words/sec' %
                              (model.epoch.eval(), current_global_step, model.learning_rate.eval(),
                               avg_loss, avg_step_time, target_words_speed))
                    else:
                        print('epoch %d global step %d lr.rate %.4f avg.loss %.4f avg. ppx %.4f avg. step time %.2f - avg. %.2f words/sec' %
                              (model.epoch.eval(), current_global_step, model.learning_rate.eval(),
                               avg_loss, ppx, avg_step_time, target_words_speed))

                if FLAGS.steps_per_checkpoint > 0:
                    if current_global_step % FLAGS.steps_per_checkpoint == 0:
                        # Save checkpoint
                        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                        saved = True

                if FLAGS.steps_per_validation > 0:

                    if current_global_step % FLAGS.steps_per_validation == 0:

                        valid_loss, valid_ppx, n_words = run_eval(
                            model=model, session=sess, data=valid_data,
                            batch_size=FLAGS.batch_size, num_steps=FLAGS.num_valid_steps
                        )

                        test_loss, test_ppx, n_words = run_eval(
                            model=model, session=sess, data=test_data,
                            batch_size=FLAGS.batch_size, num_steps=FLAGS.num_valid_steps, valid=False
                        )

                        should_stop = check_early_stop(model=model, session=sess, ppx=valid_ppx, flags=FLAGS)

                        if should_stop:
                            break

            ep = model.epoch.eval()
            print("Epoch %d finished... " % ep)

            should_stop = False

            # updating epoch number
            sess.run(model.epoch_update_op)
            ep_new = model.epoch.eval()

            if FLAGS.save_each_epoch:

                # Save checkpoint
                print("Saving current model...")
                checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

            if FLAGS.eval_after_each_epoch:

                valid_loss, valid_ppx, n_words = run_eval(
                    model=model, session=sess, data=valid_data,
                    batch_size=FLAGS.batch_size, num_steps=FLAGS.num_valid_steps
                )
                best_ppx = model.best_eval_ppx.eval()

                with codecs.open(FLAGS.best_models_dir + FLAGS.model_name + ".txt", "a", encoding="utf-8") as f:
                    f.write("PPX after epoch #%d: %f (Current best PPX: %f)\n" % (ep - 1, valid_ppx, best_ppx))

                if FLAGS.test_after_each_epoch:

                    test_loss, test_ppx, n_words = run_eval(
                        model=model, session=sess, data=test_data,
                        batch_size=FLAGS.batch_size, num_steps=FLAGS.num_valid_steps,
                        valid=False
                    )

                    with codecs.open(FLAGS.best_models_dir + FLAGS.model_name + ".txt", "a", encoding="utf-8") as f:
                        f.write("PPX after epoch #%d: %f \n" % (ep - 1, test_ppx))

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

            avg_eval_loss, avg_ppx, total_words = run_eval(model=model, session=sess, data=valid_data,
                                                           batch_size=FLAGS.batch_size,
                                                           num_steps=FLAGS.num_valid_steps)
            print('  eval: averaged valid. loss %.8f\n' % avg_eval_loss)

            print("\n##### Test Results: #####\n")

            avg_test_loss, test_ppx, total_words = run_eval(model=model, session=sess, data=test_data,
                                                            batch_size=FLAGS.batch_size,
                                                            num_steps=FLAGS.num_valid_steps, valid=False)
            print('  eval: averaged test loss %.8f\n' % avg_test_loss)

            sys.stdout.flush()


def run_eval(model, session, data, batch_size=1, num_steps=120, valid=True):
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
    if valid:
        eval_type = "Valid"
        print("\nValidating:\n")
    else:
        eval_type = "Test"
        print("\nTesting:\n")
    costs = 0.0
    iters = 0
    total_words = 0

    for step, (x, y, w, words) in enumerate(data_utils.data_iterator(data, batch_size, num_steps)):

        state = model.initial_state_train.eval()

        cost, _ = model.valid_step(session=session, lm_inputs=x, lm_targets=y, mask=w, state=state, dropout_rate=0.0)

        total_words += words
        costs += cost
        iters = step + 1
    eval = costs / iters
    ppxs = numpy.exp(eval)

    if ppxs > 10000.0:
        print("%s PPX after epoch #%d: > 10000.0 - # words %d\n" % (eval_type, model.epoch.eval(), total_words))
    else:
        print("%s PPX after epoch #%d: %f - # words %d\n" % (eval_type, model.epoch.eval(), ppxs, total_words))

    return eval, ppxs, total_words


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

    stop_patience = flags.early_stop_patience
    lr_patience = flags.lr_rate_patience

    # check early stop - if early stop patience is greater than 0, test it
    if stop_patience > 0:

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

                stop_counter = model.estop_counter.eval()

                if flags.lr_rate_patience > 0:

                    if stop_counter % lr_patience == 0 and stop_counter > 0:
                        session.run(model.learning_rate_decay_op)

                if stop_counter >= stop_patience:
                    print('\nEARLY STOP!\n')
                    stop = True

        print('\n   best valid. ppx: %.8f' % model.best_eval_ppx.eval())
        print('early stop patience: %d - max %d\n' % (int(model.estop_counter.eval()), stop_patience))

    return stop
