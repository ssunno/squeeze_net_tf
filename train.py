# -*- coding: utf-8 -*-

import tensorflow as tf
import time
from data_reader import BatchSet, load_cifar10
from squeeze_net import SqueezeNet

flags = tf.flags
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('num_epochs', 50, 'number of epochs')
flags.DEFINE_float('learning_rate', 0.04, 'init learning rate')  # TODO : add lr decay code
flags.DEFINE_float('dropout', 0.5, 'define dropout keep probability')
flags.DEFINE_float('max_grad_norm', 5.0, 'define maximum gradient normalize value')
flags.DEFINE_float('normalize_decay', 5.0, 'batch normalize decay rate')
flags.DEFINE_float('weight_decay', 0.0002, 'L2 regularizer weight decay rate')

flags.DEFINE_integer('print_every', 5, 'how often to print training status')
flags.DEFINE_string('name', None, 'name of result save dir')

FLAGS = flags.FLAGS


def train():
    # initialize data reader
    batch_set = BatchSet(FLAGS.batch_size, load_cifar10)
    print("Batch dataset initialized.\n# of training data: {}\n# of test data: {}\n# of class: {}"
          .format(len(batch_set.data_train[0]), len(batch_set.data_test[0]), batch_set.num_classes[0]))
    # initialize squeeze net model
    squeeze_net = SqueezeNet(batch_set.image_shape, batch_set.num_classes)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # tensorboard setting
        train_summary = tf.summary.merge([tf.summary.scalar("loss", squeeze_net.loss),
                                          tf.summary.scalar("accuracy", squeeze_net.accuracy)])
        test_summary = tf.summary.merge([tf.summary.scalar("loss", squeeze_net.loss),
                                         tf.summary.scalar("accuracy", squeeze_net.accuracy)])
        if FLAGS.name is None:
            FLAGS.name = time.time() % 1000
        summary_writer = tf.summary.FileWriter("./result/" + FLAGS.name, sess.graph)

        for current_epoch in range(FLAGS.num_epochs):
            # training step
            for x_batch, y_batch in batch_set.batches():
                start = time.time()
                feed = {squeeze_net.train_data: x_batch, squeeze_net.targets: y_batch,
                        squeeze_net.learning_rate: FLAGS.learning_rate, squeeze_net.dropout: FLAGS.dropout}
                _, global_step, loss, accuracy, summary = \
                    sess.run([squeeze_net.train_op, squeeze_net.global_step, squeeze_net.loss,
                              squeeze_net.accuracy, train_summary], feed_dict=feed)
                summary_writer.add_summary(summary, global_step)
                if global_step % FLAGS.print_every == 0:
                    print("{}/{} ({} epochs) step, loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
                          .format(global_step, int(round(len(batch_set.data_train[0])/FLAGS.batch_size)) * FLAGS.num_epochs, current_epoch,
                                  loss, accuracy, time.time() - start))
            # test step
            start, avg_loss, avg_accuracy = time.time(), 0, 0
            for x_test, y_test in batch_set.test_batches():
                feed = {squeeze_net.train_data: x_test, squeeze_net.targets: y_test,
                        squeeze_net.learning_rate: FLAGS.learning_rate, squeeze_net.dropout: 1.0}
                loss, accuracy, summary = sess.run([squeeze_net.loss, squeeze_net.accuracy, test_summary], feed_dict=feed)
                avg_loss += loss * len(x_test)
                avg_accuracy += accuracy * len(x_test)
                summary_writer.add_summary(summary, current_epoch)
            print("{} epochs test result. loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
                  .format(current_epoch, avg_loss / len(batch_set.data_test[0]), avg_accuracy / len(batch_set.data_test[0]), time.time() - start))


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
