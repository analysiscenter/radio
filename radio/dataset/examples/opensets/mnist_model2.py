# pylint: skip-file
import os
import sys
from time import time
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset import Pipeline, B, C, F, V, action, ImagesBatch
from dataset.opensets import MNIST
from dataset.models.tf import TFModel
from dataset.models.tf.layers import conv_block


class MyModel(TFModel):
    def _build_config(self, names=None):
        names = 'images', 'labels'
        config = super()._build_config(names)

        config['default']['data_format'] = self.data_format('images')
        config['input_block']['inputs'] = self.inputs['images']
        config['head'] = {'num_classes': self.num_classes('labels'),
                          'layout': 'cndP', 'filters': self.num_classes('labels')
                         }
        return config

    def body(self, inputs, **kwargs):
        x = conv_block(inputs, 16, 3, layout='snav snav snav', depth_multiplier=2, **kwargs)
        return x

class MyBatch(ImagesBatch):
    components = 'images', 'labels', 'digits'
    @action
    def make_digits(self):
        self.digits = (10 + self.labels).astype('str')
        return self


if __name__ == "__main__":
    BATCH_SIZE = 128

    mnist = MNIST(batch_class=MyBatch)
    config = dict(some=1, conv=dict(arg1=10))
    print()
    print("Start training...")
    t = time()
    train_tp = (Pipeline(config=config)
                .init_variable('loss_history', init_on_each_run=list)
                .init_variable('current_loss', init_on_each_run=0)
                .init_variable('input_tensor_name', 'images')
                .init_model('dynamic', MyModel, 'conv',
                            config={'session': {'config': tf.ConfigProto(allow_soft_placement=True)},
                                    'loss': 'ce',
                                    'optimizer': {'name':'Adam', 'use_locking': True},
                                    'inputs': dict(images={'shape': (None, None, 1)}, #'shape': (28, 28, 1), 'transform': 'mip @ 1'},
                                                   #labels={'shape': 10, 'dtype': 'uint8',
                                                   labels={'classes': (10+np.arange(10)).astype('str'),
                                                           'transform': 'ohe', 'name': 'targets'}),
                                    'output': dict(ops=['labels', 'accuracy'])})
                .make_digits()
                .train_model('conv', fetches='loss',
                                     feed_dict={V('input_tensor_name'): B('images'),
                                                'labels': B('digits')},
                             save_to=V('current_loss'))
                #.print_variable('current_loss')
                .update_variable('loss_history', V('current_loss'), mode='a'))

    #train_pp = (train_tp << mnist.train)
    train_pp = (train_tp << mnist.train).run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=True, prefetch=0)
    print("End training", time() - t)


    print()
    print("Start testing...")
    t = time()
    test_tp = (Pipeline()
        .import_model('conv', train_pp)
        .init_variable('accuracy', init_on_each_run=list)
        .make_digits()
        .predict_model('conv', fetches='accuracy', feed_dict={'images': B('images'),
                                                              'labels': B('digits')},
                       save_to=V('accuracy'), mode='a')
    )
    test_pp = (test_tp << mnist.test).run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=True, prefetch=0)
    print("End testing", time() - t)

    accuracy = test_pp.get_variable('accuracy')
    print('Accuracy {:6.2f}'.format(np.array(accuracy).mean()))

    for i in range(0):
        train_pp = None
        test_tp = None
        test_pp = None
        print("Start training...")
        t = time()
        train_pp = (train_tp << mnist.train)
        print('.... run', train_pp)
        train_pp.run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=True, prefetch=0)
        print("End training", time() - t)

        test_tp = None
        test_tp = (Pipeline()
            .import_model('conv', train_pp)
            .init_variable('accuracy', init_on_each_run=list)
            .make_digits()
            .predict_model('conv', fetches='accuracy', feed_dict={'images': B('images'),
                                                                  'labels': B('digits')},
                           save_to=V('accuracy'), mode='a')
        )

        print("Start testing...")
        t = time()
        test_pp = (test_tp << mnist.test)
        print('.... run', test_pp)
        test_pp.run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=True, prefetch=0)
        print("End testing", time() - t)

        accuracy = test_pp.get_variable('accuracy')
        print('                            Accuracy {:6.2f}'.format(np.array(accuracy).mean()))
