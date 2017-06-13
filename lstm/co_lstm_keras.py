#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 17-6-8 下午2:33
@Description: 采用LSTM对铜价进行预测（Keras实现）
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn import preprocessing

from common.data_loading import read_co_data_rnn
from common.model_evaluation import model_evaluation
from common.model_visualization import model_visualization


class Conf:
    # 需要加入训练的特征
    FIELDS = [
        # 交割月份
        # 'delivery_month',
        # 前结算
        'pre_settlement_price_i',
        # 今开盘
        'open_price_i',
        # 最高价-主力
        'highest_price_i',
        # 最低价-主力
        'lowest_price_i',
        # 收盘价-主力
        # 'close_price_i',
        # 结算参考价-主力
        'settlement_price_i',
        # 涨跌1
        'zd1_chg',
        # 涨跌2
        'zd2_chg',
        # 成交手-主力
        'volume_i',
        # 持仓手
        'open_interest',
        # 变化
        'open_interest_chg',

        # 综合指数(o_curproduct)
        # 最高价-综合
        'highest_price_p',
        # 最低价-综合
        'lowest_price_p',
        # 加权平均价-综合
        'avg_price_p',
        # 成交手-综合
        'volume_p',
        # 成交额(亿元)
        'turn_over',
        # 年成交手(万手)
        'year_volume',
        # 年成交额(亿元)
        'year_turn_over',

        # others 13
        # 年份
        # 'o_year',
        # 月份
        # 'o_month',
        # 日
        # 'o_day',
        # 星期
        # 'o_weekday',
        # 年期序号
        # 'o_year_num'
        # 收盘价-主力
        'close_price_i'
    ]

    # 时间序列长度
    SEQ_LEN = 50
    # epochs大小
    EPOCHS = 10
    # 批大小
    BATCH_SIZE = 500
    # 测试训练集比例
    TRAIN_SAMPLES_RATE = 0.8
    # 网络形状
    LAYERS = [len(FIELDS), SEQ_LEN, 100, 1]


def load_data():
    """
    加载铜价数据
    """
    raw_data = read_co_data_rnn()
    print(len(raw_data))
    raw_data = raw_data[['price_date'] + Conf.FIELDS].dropna()
    print(len(raw_data))

    min_date = min(raw_data.price_date)
    max_date = max(raw_data.price_date)
    date_range = pd.date_range(min_date, max_date)
    # 构造一个用于保存每天价格的DataFrame
    data = pd.DataFrame(np.full((len(date_range), len(Conf.FIELDS)), np.nan), index=date_range, columns=Conf.FIELDS)
    data.update(raw_data)
    # 采用线性插值对缺失值进行填充
    data = data.interpolate()

    # 对特征进行归一化
    min_max_scalaer = preprocessing.MinMaxScaler()
    # 将归一化后的特征更新到data中
    # data.iloc[:, :-1] = min_max_scalaer.fit_transform(data.values[::, :-1])
    # 训练数据归一化
    data = min_max_scalaer.fit_transform(data.values)

    result = []
    for i in range(len(data) - Conf.SEQ_LEN):
        result.append(data[i: i + Conf.SEQ_LEN])

    # 训练数据集数量
    train_samples_num = int(len(data) * Conf.TRAIN_SAMPLES_RATE)

    _X_train = np.array(result[:train_samples_num])
    _X_test = np.array(result[train_samples_num:])
    _y_train = np.array(data[:, -1]).T[Conf.SEQ_LEN: train_samples_num + Conf.SEQ_LEN]
    _y_test = np.array(data[:, -1]).T[train_samples_num + Conf.SEQ_LEN:]

    print(len(data), len(result), len(_X_train), len(_X_test), len(_y_train), len(_y_test))

    print(_X_train.shape)
    print(_X_test.shape)
    return [min_max_scalaer, _X_train, _y_train, _X_test, _y_test]


def build_model():
    """
    定义模型
    """
    model = Sequential()

    model.add(LSTM(units=Conf.LAYERS[1], input_shape=(Conf.LAYERS[1], Conf.LAYERS[0]), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(Conf.LAYERS[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=Conf.LAYERS[3]))
    model.add(Activation("tanh"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


def predict_by_day(model, data):
    """
    按天预测
    """
    predict = model.predict(data)
    print(predict.shape)
    predict = np.reshape(predict, (len(predict),))
    print(predict.shape)
    return predict


def inverse_trans(scaler, data):
    """
    将y反归一化
    """
    return data * ((scaler.data_max_ - scaler.data_min_)[-1]) + scaler.data_min_[-1]


if __name__ == '__main__':
    global_start_time = time.time()
    print('> Loading data... ')
    mm_scaler, X_train, y_train, X_test, y_test = load_data()
    print('> Data Loaded. Compiling...')

    rnn = build_model()
    rnn.fit(X_train, y_train, batch_size=Conf.BATCH_SIZE, epochs=Conf.EPOCHS, validation_split=0.05)

    predicted = predict_by_day(rnn, X_test)

    print('Training duration (s) : ', time.time() - global_start_time)

    predicted = inverse_trans(mm_scaler, predicted)
    y_test = inverse_trans(mm_scaler, y_test)

    # 模型评估
    model_evaluation(pd.DataFrame(predicted), pd.DataFrame(y_test), None)

    # 预测结果可视化
    model_visualization(y_test, predicted)
