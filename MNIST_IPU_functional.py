from __future__ import absolute_import, division, print_function, unicode_literals
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu

#
# Configure the IPU system
#
cfg = ipu.utils.create_ipu_config()
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)


def create_train_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()
    # x_train, y_train = x_train / 255.0, keras.utils.to_categorical(y_train)
    x_train = x_train / 255.0
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
    train_ds = train_ds.map(lambda d, l:
                            (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))
    ## ipu에서 train dataset을 구성하기 위해서 필요한 형변환 작업

    return train_ds.repeat()

def create_test_dataset():
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    # x_test, y_test = x_test / 255.0, keras.utils.to_categorical(y_test)
    x_test = x_test / 255.0
    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).shuffle(10000).batch(32, drop_remainder=True)
    test_ds = test_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))

    return test_ds.repeat()
#
# Create the model using the IPU-specific Sequential class
#
def create_model():
    m = ipu.keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    return m


# Create an IPU distribution strategy
strategy = ipu.ipu_strategy.IPUStrategy()

start = time.time()
with strategy.scope():
    # Create an instance of the model
    model = create_model()

    # Get the training dataset
    train_ds = create_train_dataset()
    test_ds = create_test_dataset()
    # Train the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    model.fit(train_ds, steps_per_epoch=1875, epochs=30)

    test_loss, test_acc = model.evaluate(test_ds, steps=2000)
end = time.time()
print(end - start)
print(test_loss, test_acc)

# accuracy : 0.9843, 경과 시간 : 38.6초
# reference : https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/examples_tf2.html#training-on-the-ipu
