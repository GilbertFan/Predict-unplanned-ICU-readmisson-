import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models,layers,preprocessing,optimizers,losses,metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics as me
import matplotlib.pyplot as plt

x_train=np.load("./data/x_t1.npy")
x_test=np.load("./data/x_v1.npy")
y_train=np.load("./data/y_t1.npy")
y_test=np.load("./data/y_v1.npy")




MAX_LEN = 61
BATCH_SIZE = 20

tf.keras.backend.clear_session()


class CnnModel(models.Model):
    def __init__(self):
        super(CnnModel, self).__init__()

    def build(self, input_shape):

        self.lstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))
        self.conv_1 = layers.Conv1D(16, kernel_size=3, name="conv_1", activation="relu")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation="sigmoid")
        super(CnnModel, self).build(input_shape)

    def call(self, x):
        x = tf.reshape(x,[-1,MAX_LEN,72])
        x = self.lstm1(x)
        print(x.shape)
        x = self.conv_1(x)
        x = self.flatten(x)
        x = self.dense(x)
        return (x)

    def summary(self):
        x_input = layers.Input(shape=MAX_LEN)
        output = self.call(x_input)
        model = tf.keras.Model(inputs=x_input, outputs=output)
        model.summary()


model = CnnModel()
model.build(input_shape=(None,MAX_LEN))
model.summary()


@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 8 + timestring)


optimizer = optimizers.Nadam()
loss_func = losses.BinaryCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.BinaryAccuracy(name='train_accuracy')

valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.BinaryAccuracy(name='valid_accuracy')


@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)
    return predictions

@tf.function
def valid_step(model, features, labels):
    predictions = model(features, training=False)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)
    return predictions

def train_model(model, x_train,y_train,x_test,y_test, epochs):
    for epoch in tf.range(1, epochs + 1):

        l_t, l_v = [], []
        p_t, p_v = [], []
        for features, labels in zip(x_train, y_train):
            labels = np.array([[labels]])
            labels = tf.reshape(labels, [20, 1])
            p = train_step(model, features, labels)
            l_t.extend(i[0] for i in labels.numpy())
            p_t.extend(i[0] for i in p.numpy())

        for features, labels in zip(x_test, y_test):
            labels = tf.reshape(labels, [20, 1])
            valid_step(model, features, labels)
            p = valid_step(model, features, labels)

            l_v.extend(i[0] for i in labels.numpy())
            p_v.extend(i[0] for i in p.numpy())

        p_t1, p_v1 = [], []
        for i in p_t:
            if i < 0.5:
                p_t1.append(0)
            else:
                p_t1.append(1)
        for i in p_v:
            if i < 0.5:
                p_v1.append(0)
            else:
                p_v1.append(1)

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'

        if epoch % 1 == 0:
            printbar()
            tf.print(tf.strings.format(logs,
                                       (epoch, train_loss.result(), train_metric.result(), valid_loss.result(),
                                        valid_metric.result())))
            tf.print("")


        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()


train_model(model, x_train,y_train,x_test,y_test,epochs=10)