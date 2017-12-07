# pylint: skip-file
import os
import sys
import threading
from time import time
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset import *
from dataset.models.tf import TFModel
from dataset.models.tf.layers import conv2d_block, flatten


class MyModel(TFModel):
    """An example of a tf model class """
    def _build(self, inputs, *args, **kwargs):
        #images_shape = self.get_from_config('images_shape', (12, 12, 1))
        #num_classes = self.get_from_config('num_classes', 3)

        #x = tf.placeholder("float", [None] + list(images_shape), name='x')
        #y = tf.placeholder("int32",[None], name='y')
        #y_oe = tf.one_hot(y, num_classes, name='targets')

        c = conv2d_block(inputs['x'], 3, 3, conv=dict(kernel_initializer=tf.contrib.layers.xavier_initializer()), max_pooling=dict(strides=4))
        f = tf.reduce_mean(c, [1,2])
        y_ = tf.identity(f, name='predictions')

        # Define a cost function
        #tf.losses.add_loss(tf.losses.softmax_cross_entropy(y_oe, y_))
        #loss = tf.losses.softmax_cross_entropy(y_oe, y_)
        #self.train_step = tf.train.AdamOptimizer().minimize(loss)
        #print(c.shape)

        print("___________________ MyModel initialized")

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        print("!=============== model loaded")


class MyBatch(Batch):
    components = 'images', 'labels'

    @action(model='static_model')
    def train_in_batch(self, model_spec):
        print("train in batch model", model_spec)
        return self

    def make_data_for_dynamic(self):
        return {'images_shape': self.images.shape, 'num_classes': 3}

    def some_method(self):
        print("some_method is called for batch", self.indices)
        return self.indices


def trans(batch):
    return dict(feed_dict=dict(x=batch.data[:, :-1], y=batch.data[:, -1].astype('int')))

# number of items in the dataset
K = 100
Q = 10


# Fill-in dataset with sample data
def gen_data():
    ix = np.arange(K)
    data = np.random.choice(255, size=(K, 12, 12, 1)).astype("float32")
    labels = np.random.choice(3, size=K).astype("int32")
    dsindex = DatasetIndex(ix)
    ds = Dataset(index=dsindex, batch_class=MyBatch)
    return ds, data, labels


# Create datasets
ds_data, data, labels = gen_data()

config = dict(dynamic_model=dict(arg1=0, arg2=0), static_model=dict(arg1=1))

# create a model
#model = MyModel()

# Create a template pipeline
pp = (Pipeline(config=config)
        .init_variable('num_classes', 3)
        .init_variable('var_name', 'num_classes')
        .init_variable('var_name2', 'loss_history3')
        .init_variable('output', init_on_each_run=list)
        .init_variable('loss_history', init_on_each_run=list)
        .init_variable('loss_history2', init_on_each_run=list)
        .init_variable('loss_history3', init_on_each_run=list)
        .init_model("static", MyModel, "static_model",
                    dict(loss='ce',
                         inputs={'x': dict(shape=(12, 12, 1)),
                                 'y': ('int32', 3, None, 'ohe', 'targets')}))
        .init_model("dynamic", MyModel, "dynamic_model",
                    dict(loss='ce',
                         inputs={'x': dict(shape=F(lambda b: b.images.shape[1:])),
                                 'y': dict(name='targets', dtype=F(lambda b: b.labels.dtype),
                                           shape=V(V('var_name')), transform='ohe')}))
#        .import_model('imported_model', model)
        #.init_model("static", TFModel, "dynamic_model2", config=dict(build=False, load=True, path='./models/dynamic'))
        .load((data, labels))
        #.train_model("static_model", fn=trans)
        .train_in_batch()
        .train_model("dynamic_model", fetches="loss", feed_dict={'x': B('images'), 'y': B('labels')},
                     save_to=V('loss_history'), mode='a')
        #.train_model("imported_model", fetches=["loss", "loss"], feed_dict={'x': B('images'), 'y': B('labels')},
        #            append_to=V('loss_history2')) #, V('loss_history3')])
        .call(MyBatch.some_method, save_to=V('output'), mode='e')
        .update_variable(V('var_name2'), [V(V('var_name'))], mode='a')
        .run(K//10, n_epochs=1, shuffle=False, drop_last=False, lazy=True)
)

# Create another template
t = time()
#res = (pp2 << ds_data).run()
print(time() - t)

print("-------------------------------------------")
print("============== start run ==================")
t = time()
res = (pp << ds_data).run()
print(time() - t)

res.save_model("dynamic_model", './models/dynamic')

m = res.get_model_by_name('dynamic_model')
m.load('./models/dynamic')

print(res.get_variable("loss_history"))

print(res.get_variable("loss_history2"))

print('output:', res.get_variable("output"))
print(res.get_variable("loss_history3"))
