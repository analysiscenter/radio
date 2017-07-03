"""Child class of CTImagesBatch that incorporates nn-models """

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append('..')
from preprocessing import CTImagesBatch, CTImagesMaskedBatch
from dataset import action, model
from .layers import vnet_down, vnet_up, deconv3d_bnorm_activation, selu
from .layers import get_dice_loss

# global constants
# input shape of a nodule
NOD_SHAPE = (32, 64, 64)

class CTImagesModels(CTImagesMaskedBatch):
    """ Сlass for describing, training nn-models of segmentation/classification;
            inference using models is also supported.

    Methods:
        1. selu_vnet_4:
            build vnet of depth = 4 using tensorflow,
            return tensors necessary for training and evaluating
    """

    @model()
    def selu_vnet_4():
        """ Describe vnet-model of depth = 4 with magic SELU activations
        Schematically:
            input(NOD_SHAPE[0], NOD_SHAPE[1], NOD_SHAPE[2], 1) -> (... // 2, 2) ->
                (... // 4, 4) -> (... // 8, 8) -> (... // 16, 16) ->
                (... // 8, 8) -> (... // 4, 4) -> (... // 2, 2) ->
                    (NOD_SHAPE[0], NOD_SHAPE[1], NOD_SHAPE[2], 1)
        Return:
            [[placeholder for input scans, p/h for input mask cancer],
             [tensor = dice loss, train step]]
        """

        # input placeholder for scan patches
        shape_inout = (None, ) + NOD_SHAPE + (1, )
        input_layer = tf.placeholder(tf.float32, shape=shape_inout,
                                     name='scans')
        training = tf.placeholder(tf.bool, shape=[], name='mode')

        # input placeholder for masks
        masks_ground_truth = tf.placeholder(tf.float32, shape=shape_inout,
                                            name='masks')

        # vnet of depth = 4
        downs = []
        ups = []

        # down 1
        net = vnet_down('down_1', input_layer, training, pool_size=[2, 2, 2],
                        strides=[2, 2, 2], channels=2, kernel=[7, 7, 7],
                        activation=selu, add_bnorm=False)
        downs.append(net)

        # down 2
        net = vnet_down('down_2', net, training, pool_size=[2, 2, 2],
                        strides=[2, 2, 2], channels=4, kernel=[5, 5, 5],
                        activation=selu, add_bnorm=False)
        downs.append(net)

        # down 3
        net = vnet_down('down_3', net, training, pool_size=[2, 2, 2],
                        strides=[2, 2, 2], channels=8, kernel=[3, 3, 3],
                        activation=selu, add_bnorm=False)
        downs.append(net)

        # down 4
        net = vnet_down('down_4', net, training, pool_size=[2, 2, 2],
                        strides=[2, 2, 2], channels=16,
                        kernel=[2, 2, 2], activation=selu, add_bnorm=False)
        downs.append(net)

        # up 1
        with tf.variable_scope('up_1'):
            net = deconv3d_bnorm_activation(net, training, kernel=[2, 2, 2],
                                           channels=8, activation=selu, add_bnorm=False)

        ups.append(net)

        # up 2
        net = vnet_up('up_2', net, downs[2], training, kernel=[3, 3, 3],
                      channels=4, activation=selu, add_bnorm=False)
        ups.append(net)

        # up 3
        net = vnet_up('up_3', net, downs[1], training, kernel=[5, 5, 5],
                      channels=2, activation=selu, add_bnorm=False)
        ups.append(net)

        # up 4
        # *Note: linear activation here
        net = vnet_up('up_4', net, downs[0], training, kernel=[7, 7, 7],
                      channels=1, activation=None, add_bnorm=False)
        ups.append(net)

        # normalize output to [0, 1]
        net = tf.nn.sigmoid(net, name='masks_predictions')

        # loss
        loss = get_dice_loss('train', net, masks_ground_truth)

        # optimization step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer().minimize(loss)

        return [[input_layer, masks_ground_truth], [loss, train_step]]


    @action(model='selu_vnet_4')
    def train_vnet(self, model, sess, verbose=False):
        """ Run iteration of training of selu_vnet_4

        model correseponding to selu_vnet_4 defined under @model decorator
            in one of methods of CTImagesModels-class

        Args:
            model: output of nn-model selu_vnet_4 returned by corresponding
                model-method of CTImagesModels-class
                *NOTE: do not supply this arg, it's always output of 
                selu_vnet_4 - method
            sess: initialized tf-session with vars that need to be updated
            verbose: whether to print stats about learning process on training
                batch
        Return:
            self
        """
        input_layer, input_masks = model[0]
        loss, train_step = model[1]

        # reshape data in batch to tensor-shape
        scans = self._data.reshape((-1, ) + NOD_SHAPE + (1, ))
        masks = self.mask.reshape((-1, ) + NOD_SHAPE + (1, ))

        # run train-step
        loss_value, _ = sess.run([loss, train_step], feed_dict={
                                 input_layer: scans, input_masks: masks})
        if verbose:
            print('current loss on train batch: ', loss_value)

        return self

    @action(model='selu_vnet_4')
    def update_test_stats(self, model, sess, stats):
        """ Compute test stats and put them into list

        Args:
            model: output of nn-model selu_vnet_4 returned by corresponding
                model-method of CTImagesModels-class
                *NOTE: do not supply this arg, it's always output of 
                selu_vnet_4 - method
            sess: initialized tf-session with vars that need to be updated
            stats: a list whith stats, in the end of which newly computed stats
                are appended
        Return:
            self

        *Note: as it is clear from the method definition, it is better to run
            this action from test subset (dataset.test.p().update_test_stats(...))
        """
        loss, _ = model[1]

        # reshape data in batch to tensor-shape
        scans = self._data.reshape((-1, ) + NOD_SHAPE + (1, ))
        masks = self.mask.reshape((-1, ) + NOD_SHAPE + (1, ))

        # run loss-op on data from batch
        loss_value = sess.run(loss, feed_dict={input_layer: scans, 
                                               input_masks: masks})

        # add computed number to list of stats:
        stats.append(loss_value)

        return self






