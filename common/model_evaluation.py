#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/15 14:20
@Description: 模型评估
"""

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, \
    r2_score


def model_evaluation(actual, predict, fed_data):
    print("explained_variance_score: " + str(explained_variance_score(actual, predict)))
    print("mean_absolute_error: " + str(mean_absolute_error(actual, predict)))
    print("mean_squared_error: " + str(mean_squared_error(actual, predict)))
    print("median_absolute_error: " + str(median_absolute_error(actual, predict)))
    print("r2_score: " + str(r2_score(actual, predict)))

    # 趋势正确性评估（价格预测准确率高不等价于趋势预测正确率高，取消该指标）
    # _calc_trend_accuracy(predict, fed_data)

    # 平均相对误差（Mean Relative Error）
    _calc_mre(actual, predict)


def _calc_mre(actual, predict):
    """
    计算平均相对误差（Mean Relative Error）
    """
    print("sum relative error: " + str(((actual - predict) / predict).sum().values))
    print("mean relative error: " + str(((actual - predict) / predict).mean().values))


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
