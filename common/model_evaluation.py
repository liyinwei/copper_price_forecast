#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/15 14:20
@Description: 模型评估
"""

import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score


def model_evaluation(y_true, y_pred):
    metrics = _cal_metrics(y_true, y_pred)
    for (k, v) in metrics.items():
        print(k + ": " + str(v))


def model_evaluation_multi_step(y_true, y_pred):
    print(y_true.shape, y_pred.shape)
    # y_true.shape = (1010, 20)
    metrics_list = []
    # 分别求t + 1, t + 2, ... , t + STEP_LEN 每天的评估指标
    for i in range(len(y_true.T)):
        metrics = _cal_metrics(pd.DataFrame(y_true.T[i]), pd.DataFrame(y_pred.T[i]))
        metrics_list.append(metrics)
        print("-------------------------------------------------------------------------")
        print("metrics of t + " + str(i + 1))
        print("-------------------------------------------------------------------------")
        for (k, v) in metrics.items():
            print("\r" + k + ": " + str(v))

    # 求总体平均
    for i in range(1, len(metrics_list)):
        for (k, v) in metrics_list[0].items():
            metrics_list[0][k] += metrics_list[i][k]
    print("-------------------------------------------------------------------------")
    print("metrics of avg")
    print("-------------------------------------------------------------------------")
    for (k, v) in metrics_list[0].items():
        print("\r" + k + ": " + str(float(v) / y_true.shape[1]))


def _cal_metrics(y_true, y_pred):
    """
    计算各个指标的值
    """
    re = _calc_re(y_true, y_pred)
    metrics = {
        "explained_variance_score":
            explained_variance_score(y_true, y_pred),
        "mean_absolute_error":
            mean_absolute_error(y_true, y_pred),
        "mean_squared_error":
            mean_squared_error(y_true, y_pred),
        "median_absolute_error":
            median_absolute_error(y_true, y_pred),
        "r2_score":
            r2_score(y_true, y_pred),
        "sum_relative_error":
            re[0],
        "mean_relative_error":
            re[1]
    }

    return metrics


def _calc_re(y_true, y_pred):
    """
    计算相对误差（Sum/Mean Relative Error）
    """
    return [((y_true - y_pred) / y_pred).sum().values, ((y_true - y_pred) / y_pred).mean().values]


def _calc_trend_accuracy(predict, fed_data):
    """
    趋势正确性评估，即判断当前收盘价与前一天收盘价对比上升/下降趋势是否正确
    """
    # 全局索引
    global_index = fed_data.index
    # 预测的样本总数
    predict_sample_no = predict.size
    # 预测结果趋势正确的样本总数
    correct_trend_no = 0
    for index, pre in predict.iterrows():
        # 获取前一天记录的索引值
        pre_index = global_index.get_values()[global_index.get_loc(index) - 1]
        # 获取前一天的收盘价
        v_pre = fed_data.loc[pre_index]['close_price_i']
        # 获取当前的收盘价
        v_target = fed_data.loc[index]['close_price_i']
        # 当天收盘价的预测值
        v_predict = pre[0]
        if (v_target - v_pre) * (v_predict - v_pre) > 0:
            correct_trend_no += 1
    print("trend accuracy rate: " + str(correct_trend_no / predict_sample_no))


if __name__ == '__main__':
    pass
