# pylint: skip-file
import os
import sys
from time import time
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset import Pipeline, B, C, F, V
from dataset.opensets import MNIST
from dataset.models.tf import FCN32, FCN16, FCN8, LinkNet, UNet, VNet


def make_masks(batch, *args):
    masks = np.zeros_like(batch.images)
    coords = np.where(batch.images > 0)
    masks[coords] = batch.labels[coords[0]]
    return np.squeeze(masks)

def make3d_images(batch, *args):
    images = np.concatenate([batch.images] * 14, axis=3)
    return np.expand_dims(images, axis=4)

def make3d_masks(batch, *args):
    images = np.concatenate([batch.images] * 14, axis=3)
    masks = np.zeros_like(images)
    coords = np.where(images > 0)
    masks[coords] = batch.labels[coords[0]]
    return np.squeeze(masks)


if __name__ == "__main__":
    BATCH_SIZE = 16

    mnist = MNIST()

    train_template = (Pipeline()
                .init_variable('loss_history', init_on_each_run=list)
                .init_variable('current_loss', init_on_each_run=0)
                .init_variable('pred_label', init_on_each_run=list)
                .init_model('dynamic', VNet, 'conv',
                            config={'loss': 'ce',
                                    'optimizer': {'name':'Adam', 'use_locking': True},
                                    'inputs': dict(images={'shape': (None, None, None, 1)},
                                                   masks={'shape': (None, None, None), 'classes': 10, 'transform': 'ohe', 'name': 'targets'}),
                                    #'input_block': {'filters': 16}
                                    'input_block/inputs': 'images'
                                    })
                                    #'output': dict(ops=['labels', 'accuracy'])})
                .train_model('conv', fetches='loss',
                                     feed_dict={'images': F(make3d_images), #B('images'),
                                                'masks': F(make3d_masks)},
                             save_to=V('current_loss'))
                .print_variable('current_loss')
                .update_variable('loss_history', V('current_loss'), mode='a'))

    train_pp = (train_template << mnist.train)
    print("Start training...")
    t = time()
    train_pp.run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=False, prefetch=0)
    print("End training", time() - t)


    print()
    print("Start testing...")
    t = time()
    test_pp = (mnist.test.p
                .import_model('conv', train_pp)
                .init_variable('accuracy', init_on_each_run=list)
                .predict_model('conv', fetches='accuracy', feed_dict={'images': B('images'),
                                                                      'masks': F(make_masks)},
                               save_to=V('accuracy'), mode='a')
                .run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=True, prefetch=0))
    print("End testing", time() - t)

    accuracy = np.array(test_pp.get_variable('accuracy')).mean()
    print('Accuracy {:6.2f}'.format(accuracy))


    conv = train_pp.get_model_by_name("conv")
