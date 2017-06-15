#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/6/8 20:01
@Description: 采用LSTM进行sin函数、股票（标准普尔500股权指数）及期铜预测
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from common.model_evaluation import model_evaluation


class Conf:
    # epochs
    EPOCHS = 1
    # 时间序列长度
    SEQ_LEN = 50
    # 预测步数
    PREDICT_STEP = 20
    # 测试训练集比例
    TRAIN_DATA_RATE = 0.9
    # 批大小
    BATCH_SIZE = 500
    # 网络形状
    LAYERS = [1, 50, 100, 1]


def load_data(filename):
    """
    数据准备
    """
    data = pd.read_csv(filename).values

    result = []
    for index in range(len(data) - Conf.SEQ_LEN - 1):
        result.append(data[index: index + Conf.SEQ_LEN + 1])
    # 数据标准化
    result = normalise_windows(result)

    result = np.array(result)

    row = round(result.shape[0] * Conf.TRAIN_DATA_RATE)
    train = result[:int(row), :]
    np.random.shuffle(train)

    _X_train = train[:, :-1]
    _y_train = train[:, -1]
    _X_test = result[int(row):, :-1]
    _y_test = result[int(row):, -1]

    # 增加一列
    _X_train = _X_train[:, :, np.newaxis]
    _X_test = _X_test[:, :, np.newaxis]

    print(_X_train.shape)
    print(_X_test.shape)
    return [_X_train, _y_train, _X_test, _y_test]


def normalise_windows(window_data):
    """
    对原始数据做标准化：n_i = (p_i/p)0 - 1)
    对应的反标准化公式为：p_i = p_0(n_i + 1)
    """
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def build_model(layers):
    """
    模型定义
    """
    model = Sequential()

    model.add(LSTM(units=layers[1], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=layers[3]))
    model.add(Activation("tanh"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    """
    每次预测1步
    """
    predict = model.predict(data)
    print(predict.shape)
    predict = np.reshape(predict, (len(predict),))
    print(predict.shape)
    return predict


def predict_sequences_multiple(model, data, window_size, prediction_len):
    """
    每次预测Conf.SEQ_LEN步
    """
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def predict_sequence_full(model, data, window_size):
    """
    每次预测所有步
    """
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    global_start_time = time.time()

    print('> Loading data... ')

    # sin: sin.csv; stock: stock.csv; copper: co.csv
    X_train, y_train, X_test, y_test = load_data('co.csv')

    print('> Data Loaded. Compiling...')

    model = build_model(Conf.LAYERS)

    model.fit(X_train, y_train, batch_size=Conf.BATCH_SIZE, epochs=Conf.EPOCHS, validation_split=0.05)

    # 预测一步
    # predicted = predict_point_by_point(model, X_test)
    # 预测所有步
    # predicted = predict_sequence_full(model, X_test, Conf.SEQ_LEN)
    # 预测Conf.SEQ_LEN步
    predicted = predict_sequences_multiple(model, X_test, Conf.SEQ_LEN, 50)

    print('Training duration (s) : ', time.time() - global_start_time)

    # 预测一步及所有步
    # plot_results(predicted, y_test)
    # 预测Conf.SEQ_LEN步
    plot_results_multiple(predicted, y_test, Conf.SEQ_LEN)

    model_evaluation(pd.DataFrame(y_test), pd.DataFrame(predicted), None)


if __name__ == '__main__':
    main()
