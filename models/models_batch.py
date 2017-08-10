"""Child class of CTImagesBatch that incorporates nn-models """

import tensorflow as tf

from ..preprocessing import CTImagesMaskedBatch
from ..dataset import action, model
from .layers import vnet_down, vnet_up, deconv3d_bnorm_activation, selu
from .layers import tf_dice_loss

from .keras_unet import KerasUnet
from .keras_resnet import KerasResNet
from .keras_vgg16 import KerasVGG16
from .keras_model import KerasModel
# global constants
# input shape of a nodule
NOD_SHAPE = (32, 64, 64)

PRETRAINED_UNET_PATH = ''
PRETRAINED_RESNET_PATH = ''
PRETRAINED_VGG16_PATH = ''

class CTImagesModels(CTImagesMaskedBatch):
    """ Сlass for describing, training nn-models of segmentation/classification;
            inference using models is also supported.

    Methods:
        1. selu_vnet_4:
            build vnet of depth = 4 using tensorflow,
            return tensors necessary for training, evaluating and inferencing
        2. train_vnet_4:
            train selu_vnet_4 on images and masks, that are contained in batch
        3. update_test_stats:
            method for evaluation of the model on test-batch (test-dataset) during
            training
        4. get_cancer_segmentation:
            method for performing inference on images of batch using trained model
    """

    @model()
    def unet():
        """ Get unet model implemented in keras. """
        return KerasUnet('unet')

    @model()
    def unet_pretrained():
        """ Get pretrained keras unet model. """
        pretrained_unet = KerasUnet('pretrained_unet')
        pretrained_unet.load_model(PRETRAINED_UNET_PATH)
        pretrained_unet.compile()
        return pretrained_unet

    @model()
    def resnet():
        """ Get resnet model implemented in keras. """
        model = KerasResNet('resnet')
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    @model()
    def resnet_pretrained():
        """ Get pretrained resnet model. """
        model = KerasModel('pretrained_resnet')
        model.load(PRETRAINED_RESNET_PATH)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    @model()
    def vgg16():
        """ Get vgg16 model implemented in keras. """
        model = KerasVGG16('vgg16')
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    @model
    def vgg16_pretrained():
        """ Get pretrained vgg16 model. """
        model = KerasModel('pretrained_vgg16')
        model.load(PRETRAINED_VGG16_PATH)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    @action
    def classification_train(self, model_name):
        """ Train model for classification task.

        Args:
        - model_name: str, name of the model;

        Returns:
        - self, unchanged batch;
        """
        model = self.get_model_by_name(model_name)
        x = np.zeros((len(self), 32, 64, 64), dtype=np.float)
        y = np.zeros(len(self), dtype=np.float)
        for i in range(len(self)):
            _x, _y = self.get(i, 'images'), self.get(i, 'masks')
            x[i, ...] = _x
            y[i] = int(np.sum(_y) > 10)
        model.train_on_batch(x[..., np.newaxis], y[: , np.newaxis])
        return self

    @action
    def classification_predict_on_crop(self, model_name, dst_dict):
        """ Get predictions on crops of model trained for classification task.

        Args:
        - model_name: str, name of the model;
        - dst_dict: dict, this dict is updated with model's predictions;

        Returns:
        - self, unchanged batch;
        """
        model = self.get_model_by_name(model_name)
        x = np.zeros((len(self), 32, 64, 64), dtype=np.float)
        for i in range(len(self)):
            x[i, ...] = self.get(i, 'images')
        predictions = model.predict_on_batch(x)
        dst_dict.update(zip(self.indices, predictions))
        return self

    @action
    def segmentation_train(self, model_name):
        """ Train model for segmentation task.

        Args:
        - model_name: str, name of the model;

        Returns:
        - self, unchanged batch;
        """
        model = self.get_model_by_name(model_name)
        x = np.zeros((len(self), 32, 64, 64), dtype=np.float)
        y = np.zeros((len(self), 32, 64, 64), dtype=np.float)
        for i in range(len(self)):
            x[i, ...], y[i, ...] = self.get(i, 'images'), self.get(i, 'masks')
        model.train_on_batch(x[: , np.newaxis, ...], y[: , np.newaxis, ...])
        y_pred = model.predict_on_batch(x[: , np.newaxis, ...])
        print("Dice on train: ", dice(y_pred, y))
        return self


    @action
    def segmentation_predict_on_crop(self, model_name, dst_dict):
        """ Get predictions on crops of model trained for segmentation task. """
        model = self.get_model_by_name(model_name)
        x = np.zeros((len(self), 32, 64, 64))
        for i in range(len(self)):
            x[i, ...] = self.get(i, 'images')
        predictions = model.predict_on_batch(x[:, np.newaxis, ...])
        dst_dict.update(zip(self.indices, predictions))
        return self

    @action
    def segmentation_predict_patient(self, model_name, strides=(16, 32, 32), batch_size=20):
        """ Get predictions on patient of model trained for segmentation task.

        Args:
        - model_name: str, name of model;
        - strides: tuple(int) of size 3, strides
        for get_patches and load_from_patches methods;
        - batch_size: int, size of batch to use for feeding data in model;

        Returns:
        - self, batched with predicted masks loaded;
        """
        model = self.get_model_by_name(model_name)
        patches_arr = self.get_patches(patch_shape=(32, 64, 64), stride=strides, padding='reflect')
        patches_arr = patches_arr.reshape(-1, 32, 64, 64)
        patches_arr = patches_arr[: , np.newaxis, ...]

        predictions = []
        for i in range(0, patches_arr.shape[0], batch_size):
            patch_mask = model.predict_on_batch(data[i: i + 20])
            predictions.append(patch_mask)

        self.load_from_patches(stride=strides,
                               scan_shape=(self.images_shape[0, :]),
                               data_attr='masks')
        return self

    @action
    def classification_test_on_dataset(self, model_name, dataset, callbacks=None):
        """ Get predictions on crops of model trained for classification task.

        Args:
        - model_name: str, name of the model;
        - dataset: Dataset containing test samples;
        - callbacks: list of callables for evaluation of metrics
        on test dataset(each callable corresponds to metric);

        Retruns:
        - self, unchanged batch;
        """
        model = self.get_model_by_name(model_name)
        for batch in dataset.gen_batch(8):
            batch = batch.load(fmt='blosc')

            x = np.zeros((len(batch), 32, 64, 64), dtype=np.float)
            y = np.zeros(len(batch), dtype=np.float)
            for i in range(len(batch)):
                _x, _y = batch.get(i, 'images'), batch.get(i, 'masks')
                x[i, ...] = _x
                y[i] = int(np.sum(_y) > 10)
            model.test_on_batch(x[..., np.newaxis], y[:, np.newaxis])
        return self

    @action
    def segmentation_test_on_dataset(self, model_name, dataset, callbacks=None):
        """ Get predictions on crops of model trained for segmentation task.

        Args:
        - model_name: str, name of the model;
        - dataset: Dataset containing test samples;
        - callbacks: list of callables for evaluation of metrics
        on test dataset(each callable corresponds to metric);

        Retruns:
        - self, unchanged batch;
        """
        model = self.get_model_by_name(model_name)
        for batch in dataset.gen_batch(8):
            batch = batch.load(fmt='blosc')

            x = np.zeros((len(batch), 32, 64, 64), dtype=np.float)
            y = np.zeros((len(batch), 32, 64, 64), dtype=np.float)
            for i in range(len(batch)):
                 x[i, ...], y[i, ...] = self.get(i, 'images'), self.get(i, 'masks')
            model.test_on_batch(x[: , np.newaxis, ...], y[:, np.newaxis, ...])
        return self

    @model()
    def selu_vnet_4(): # pylint: disable=no-method-argument
        """ Describe vnet-model of depth = 4 with magic SELU activations
        Schematically:
            input(NOD_SHAPE[0], NOD_SHAPE[1], NOD_SHAPE[2], 1) -> (... // 2, 2) ->
                (... // 4, 4) -> (... // 8, 8) -> (... // 16, 16) ->
                (... // 8, 8) -> (... // 4, 4) -> (... // 2, 2) ->
                    (NOD_SHAPE[0], NOD_SHAPE[1], NOD_SHAPE[2], 1)
        Return:
            [[placeholder for input scans, p/h for input mask cancer],
             [dice loss, train step, predicted cancer masks]]
        """

        # input placeholder for scan patches in skyscraper-form
        sky_shape = (None, ) + NOD_SHAPE[1:]
        input_layer = tf.placeholder(tf.float32, shape=sky_shape, name='scans')

        # input placeholder for masks in skyscraper-form
        masks_ground_truth = tf.placeholder(tf.float32, shape=sky_shape, name='masks')

        # input placeholder for phase-variable (e.g. needed for nets with batch-norm)
        training = tf.placeholder(tf.bool, shape=[], name='mode')

        # reshape inputs to tensor shape
        t_shape = (-1, ) + NOD_SHAPE + (1, )
        input_tshaped = tf.reshape(input_layer, t_shape)
        masks_tshaped = tf.reshape(masks_ground_truth, t_shape)


        # vnet of depth = 4
        downs = []
        ups = []

        # down 1
        net = vnet_down('down_1', input_tshaped, training, pool_size=[2, 2, 2],
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
        with tf.variable_scope('up_1'):                                                 # pylint: disable=not-context-manager
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

        # loss computed on t-shaped data
        loss = tf_dice_loss('train', net, masks_tshaped)

        # reshape vnet-output to skyscraper-shape
        sky_shape = (-1, ) + NOD_SHAPE[1:]
        net = tf.reshape(net, sky_shape)

        # optimization step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer().minimize(loss)

        return [[input_layer, masks_ground_truth], [loss, train_step, net]]


    @action(model='selu_vnet_4')
    def train_vnet(self, model_outs, sess, verbose=False):
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
        input_layer, input_masks = model_outs[0]
        loss, train_step, _ = model_outs[1]

        # run train-step
        loss_value, _ = sess.run([loss, train_step], feed_dict={
            input_layer: self.images, input_masks: self.masks})
        if verbose:
            print('current loss on train batch: ', loss_value)

        return self

    @action(model='selu_vnet_4')
    def update_test_stats(self, model_outs, sess, stats):
        """ Compute test stats and put them into list

        Args:
            model: output of nn-model selu_vnet_4 returned by corresponding
                model-method of CTImagesModels-class
                *NOTE: do not supply this arg, it's always output of
                selu_vnet_4 - method
            sess: tf-session with trained (to an extent) weights
            stats: a list whith stats, in the end of which newly computed stats
                are appended
        Return:
            self

        *NOTE: as it is clear from the method definition, it is better to run
            this action from test subset:

            dataset.test.p().update_test_stats(...)

        *NOTE: running this from a pipeline on batches of large size can be
            time-consuming. Might be better to run the action directly from
            precomputed test-batch:

            ind = FilesIndex(...)
            ds = Dataset(index=ind, batch_class=CTImagesModels)
            ds.cv_split([0.8, 0.2])
            testflow = ds.test.p.load(...).load_mask()

            # generate large batch
            testbatch = testflow.next_batch(batch_size=500)

            losses = []
            # execute the action inside training cycle:
            for i in range(num_iter):
                testbatch.update_test_stats(sess, losses)

        """
        input_layer, input_masks = model_outs[0]
        loss, _, _ = model_outs[1]

        # run loss-op on data from batch
        loss_value = sess.run(loss, feed_dict={input_layer: self.images, input_masks: self.masks})

        # add computed number to list of stats:
        stats.append(loss_value)

        return self

    @action(model='selu_vnet_4')
    def get_cancer_segmentation(self, model_outs, sess, result):
        """ Get cancer segmentation using trained weights stored in
                tf-session sess

        Args:
            model: output of nn-model selu_vnet_4 returned by corresponding
                model-method of CTImagesModels-class
                *NOTE: do not supply this arg, it's always output of
                selu_vnet_4 - method
            sess: tf-session with trained weights
            result: 3d-array for storing obtained segmentation
                should have skyscraper-shape =
                (len(self) * NOD_SHAPE[0], NOD_SHAPE[1], NOD_SHAPE[2])

        Return:
            self
        """
        input_layer, _ = model_outs[0]
        _, _, masks_predictions = model_outs[1]

        # compute predicted masks
        predictions = sess.run(masks_predictions, feed_dict={input_layer: self.images})

        # put the masks into result-array
        result[:, :, :] = predictions

        return self
