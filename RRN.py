# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import sys
from DataHelper import Data


def main():
    model = RRN()
    model.run()


class RRN:
    def __init__(self):
        # params parser
        self.batch_size = 50
        self.n_step = 1
        self.lr = 0.01
        self.verbose = 10
        # Data
        dataSet = Data("ml-1m")
        self.train = dataSet.data.values
        # Model
        self.add_placeholder()
        self.add_embedding_layer()
        self.add_rnn_layer()
        self.add_pred_layer()
        self.add_loss()
        self.add_train_step()
        self.init_session()

    def add_placeholder(self):
        # user placeholder
        self.userID = tf.placeholder(tf.int32, shape=[None, 1], name="userID")
        # movie placeholder
        self.movieID = tf.placeholder(tf.int32, shape=[None, 1], name="movieID")
        # target
        self.rating = tf.placeholder(tf.float32, shape=[None, 1], name="rating")
        # other params
        self.dropout = tf.placeholder(tf.float32, name='dropout')

    def add_embedding_layer(self):
        with tf.name_scope("userID_embedding"):
            # user id embedding
            uid_onehot = tf.reshape(tf.one_hot(self.userID, 6040), shape=[-1, 6040])
            # uid_onehot_rating = tf.multiply(self.rating, uid_onehot)
            uid_layer = tf.layers.dense(uid_onehot, units=128, activation=tf.nn.relu)
            self.uid_layer = tf.reshape(uid_layer, [-1, self.n_step, 128])

        with tf.name_scope("movie_embedding"):
            # movie id embedding
            mid_onehot = tf.reshape(tf.one_hot(self.movieID, 3952), shape=[-1, 3952])
            # mid_onehot_rating = tf.multiply(self.rating, mid_onehot)
            mid_layer = tf.layers.dense(mid_onehot, units=128, activation=tf.nn.relu)
            self.mid_layer = tf.reshape(mid_layer, shape=[-1, self.n_step, 128])

    def add_rnn_layer(self):
        with tf.variable_scope("user_rnn_cell"):
            userCell = tf.nn.rnn_cell.GRUCell(num_units=128)

            userInput = tf.transpose(self.mid_layer, [1, 0, 2])
            # userInput = tf.reshape(userInput, [-1, 128])
            # userInput = tf.split(userInput, self.n_step, axis=0)

            userOutputs, userStates = tf.nn.dynamic_rnn(userCell, userInput, dtype=tf.float32)
            self.userOutput = userOutputs[-1]
        with tf.variable_scope("movie_rnn_cell"):
            movieCell = tf.nn.rnn_cell.GRUCell(num_units=128)

            movieInput = tf.transpose(self.uid_layer, [1, 0, 2])
            movieOutputs, movieStates = tf.nn.dynamic_rnn(movieCell, movieInput, dtype=tf.float32)
            self.movieOutput = movieOutputs[-1]

    def add_pred_layer(self):
        W = {
            'userOutput': tf.Variable(tf.random_normal(shape=[128, 64], stddev=0.1)),
            'movieOutput': tf.Variable(tf.random_normal(shape=[128, 64], stddev=0.1))
        }
        b = {
            'userOutput': tf.Variable(tf.random_normal(shape=[64], stddev=0.1)),
            'movieOutput': tf.Variable(tf.random_normal(shape=[64], stddev=0.1))
        }
        userVector = tf.add(tf.matmul(self.userOutput, W['userOutput']), b['userOutput'])
        movieVector = tf.add(tf.matmul(self.movieOutput, W['movieOutput']), b['movieOutput'])

        self.pred = tf.reduce_sum(tf.multiply(userVector, movieVector), axis=1, keep_dims=True)

    def add_loss(self):
        losses = tf.losses.mean_squared_error(self.rating, self.pred)
        self.loss = tf.reduce_mean(losses)

    def add_train_step(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def init_session(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

    def run(self):
        length = len(self.train)
        batches = length // self.batch_size + 1

        # shuffled_idx = np.random.permutation(np.arange(len(self.train)))
        # self.train = self.train[shuffled_idx]

        train_loss = []
        for i in range(batches):
            minIdx = i * self.batch_size
            maxIdx = min(length, (i+1)*self.batch_size)
            train_batch = self.train[minIdx:maxIdx]
            feed_dict = self.createFeedDict(train_batch)

            tmpLoss = self.sess.run(self.loss, feed_dict=feed_dict)
            train_loss.append(tmpLoss)

            self.sess.run(self.train_op, feed_dict=feed_dict)

            if self.verbose and i % self.verbose == 0:
                sys.stdout.write('\r{} / {}： loss = {}'.format(
                    i, batches, np.sqrt(np.mean(train_loss[-20:]))
                ))
                sys.stdout.flush()
        print("Training Finish, Last 2000 batches loss is {}.".format(
            np.sqrt(np.mean(train_loss[-2000:]))
        ))

    def createFeedDict(self, data, dropout=1.):
        userID = []
        movieID = []
        ratings = []
        for i in data:
            userID.append([i[0]-1])
            movieID.append([i[1]-1])
            ratings.append([float(i[2])])
        return {
            self.userID: np.array(userID),
            self.movieID: np.array(movieID),
            self.rating: np.array(ratings),
            self.dropout: dropout
        }


if __name__ == '__main__':
    main()
