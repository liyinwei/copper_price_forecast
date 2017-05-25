#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/4/25 9:08
@Description: 采用MLP进行铜价预测
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from data_loading import read_co_data
from feature_engineering import feature_engineering
from model_evaluation import model_evaluation
from model_visualization import model_visualization

if __name__ == '__main__':
    # 读取原始数据
    raw_data = read_co_data()
    # 特征工程
    fed_data = feature_engineering(raw_data)
    # feature vector
    X = fed_data.take(list(range(fed_data.shape[1] - 1)), axis=1)
    # target
    y = np.ravel(fed_data.take([fed_data.shape[1] - 1], axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # 定义一个BP神经网络
    reg = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    # 训练
    print("start training...")
    reg.fit(X_train, y_train)
    print("training finished.")
    # 预测
    print("start predicting...")
    y_pred = reg.predict(X_test)
    print("predicting finished.")

    y_pred = pd.DataFrame(y_pred)
    y_pred.index = X_test.index

    y_test = pd.DataFrame(y_test)
    y_test.index = X_test.index
    # 将结果写入文件
    # pd.DataFrame(y_pred).to_excel('y_pred.xlsx')
    # 模型评估
    model_evaluation(y_test, y_pred, fed_data)
    # 可视化
    model_visualization(y_test, y_pred)

    print(type(X), type(y), type(X_train), type(X_test), type(y_train), type(y_test), type(y_pred))
