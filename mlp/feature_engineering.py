#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/15 14:14
@Description: 特征工程
"""

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA


def feature_engineering(df):
    """
    特征工程
    """
    print("start feature enginnring...")
    # 数据预处理
    df = _pre_process(df).copy()
    # 特征构建
    df = _feature_construction(df)
    # 删除多余的历史序列(n_sample = max(max(_get_history_statistic_days(), _get_history_days())))
    df = _delete_redundant_samples(df)
    # 特征选择
    df = _feature_selection(df)
    # 归一化处理
    df = _pp_min_max_scale(df)
    # 降维
    df = _pca_decomposition(df)
    print("feature enginnring finished")
    return df


def _pre_process(df):
    """
    数据预处理
    """
    print("  start pre_processing...")
    # 将星期转换为数值
    df = _pp_weekday(df)
    # 处理settlement_price_m异常值（空）
    df = _pp_settlement_price_m(df)
    print("  pre_processing finished.")
    return df


def _feature_construction(df):
    """
    特征构建
    """
    print("  start feature construction...")
    df = _fc_trend(df)
    df = _fc_history_statistic(df)
    df = _fc_history_series(df)
    print("  feature construction finished.")
    return df


def _feature_selection(df):
    """
    特征选择
    """
    print("  start feature selection...")
    label_column = 'close_price_i'
    features = _get_selected_features()
    features.append(label_column)
    print(len(features))
    print("  feature selection finished.")
    return df[features]


def _pca_decomposition(df):
    """
    利用PCA进行降维
    """
    print("  start pca decomposition...")
    # df.to_excel('before_pca.xlsx')
    # 保存index信息
    index = df.index
    # 保存target列
    target = np.array(df.iloc[:, -1])
    target.shape = (len(target), 1)

    pca = PCA(n_components=30)
    # pca = KernelPCA(kernel='rbf')
    df = pca.fit_transform(df.iloc[:, :-1])

    # 合并pca后的主成份和target
    df = pd.DataFrame(np.hstack((df, target)))
    # 重新索引
    df.index = index
    # df.to_excel('after_pca.xlsx')
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.cumsum())

    print("  pca decomposition finished.")
    return df


def _delete_redundant_samples(df):
    """
    删除多余的历史序列(n_sample = max(max(_get_history_statistic_days(), _get_history_days())))
    """
    print("  start deleting redundant samples...")
    days = _get_history_statistic_days()
    days.append(_get_history_days())
    print("  deleting redundant samples finished.")
    return df.iloc[max(days):]


def _pp_min_max_scale(df):
    """
    特征值归一化处理
    """
    print("  start minmax scaling...")
    # drop掉id和price_date字段
    # df = df.drop(['id', 'price_date'], axis=1)
    # 保存index信息及column信息
    index = df.index
    columns = df.columns
    # 对特征进行归一化
    feature_scaled = preprocessing.MinMaxScaler().fit_transform(df.iloc[:, :-1])

    target = np.array(df.iloc[:, -1])
    target.shape = (len(target), 1)

    # 合并归一化后的X和未做归一化的y（归一化后Pandas 的 DataFrame类型会转换成numpy的ndarray类型）
    df_scaled = pd.DataFrame(np.hstack((feature_scaled, target)))
    # 重新设置索引及column信息
    df_scaled.index = index
    df_scaled.columns = columns

    print("  minmax scaling finished.")
    return df_scaled


def _pp_weekday(df):
    """
    Pre_Process: 将星期中文替换为数值
    """
    df['o_weekday'] = df.o_weekday.map({'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '日': 7}).astype(int)
    return df


def _pp_settlement_price_m(df):
    """
    Pre_Process: 过滤settlement_price_m字段为空的记录
    """
    return df.dropna()


def _fc_trend(df):
    """
    构建历史趋势特征
    """
    print("    start trend feature construction...")
    history_statistic_days = _get_history_statistic_days()
    df = _get_trend(df, history_statistic_days)
    print("    trend feature construction finished.")
    return df


def _fc_history_statistic(df):
    """
    构建过去n天均值作为特征，一方面在特征中加入历史影响，另一方面避免用当天特征预测当天价格的情况
    """
    print("    start history feature construction...")
    history_statistic_days = _get_history_statistic_days()
    df = _get_history_avg(df, history_statistic_days)
    print("    history feature construction finished.")
    return df


def _fc_history_series(df):
    """
    构建历史序列特征
    """
    print("    start series feature construction...")
    window_size = _get_history_days()
    df = _get_history_series(df, window_size)
    print("    series feature construction finished.")
    return df


def _get_history_statistic_days():
    """
    获取历史数据统计的时间跨度（天/交易日）
    """
    return [1, 3]


def _get_history_days():
    """
    获取时间序列窗口天数（天 / 交易日）
    """
    return 5


def _get_history_features():
    """
    获取需要提取历史统计数据的特征
    """
    features = [  # 'id', 'price_date', 'product_id', 'product_sort_no', 'product_name', 'delivery_month',
        'pre_settlement_price_i', 'open_price_i', 'highest_price_i', 'lowest_price_i',
        'close_price_i', 'settlement_price_i', 'zd1_chg', 'zd2_chg', 'volume_i',
        'open_interest', 'open_interest_chg',
        # 'order_no',
        'highest_price_p', 'lowest_price_p',
        'avg_price_p', 'volume_p', 'turn_over', 'year_volume', 'year_turn_over',
        # 'trading_day',
        'last_price', 'open_price_m', 'close_price_m', 'pre_close_price_m',
        'updown', 'updown1', 'updown2', 'highest_price_m', 'lowest_price_m',
        'avg_price_m', 'settlement_price_m']
    # 'o_year', 'o_month', 'o_day', 'o_weekday', 'o_year_num', 'o_total_num', 'o_trade_day'
    # 'o_imchange_data', 'o_code', 'o_msg', 'report_date', 'update_date', 'print_date'
    return features


def _get_common_features():
    """
    原始数据中需要参与预测但不参与构建历史统计特征的特征集合
    """
    features = ['o_year', 'o_month', 'o_day', 'o_weekday', 'o_year_num']
    return features


def _get_updown_features():
    """
    涨跌特征 
    """
    return ['zd1_chg']


def _get_trend(df, history_statistic_days):
    """
    构建历史趋势信息
    """
    updown_features = _get_updown_features()
    for index, row in df.iterrows():
        for day in history_statistic_days:
            if index[0] < day:
                continue
            for feature in updown_features:
                df.set_value(index[0], feature + '_trend_' + str(day), df[index[0] - day: index[0]][feature].sum())
    return df


def _get_history_avg(df, history_statistic_days):
    """
    构建时间序列历史信息特征（统计类）
    """
    features = _get_history_features()
    for index, row in df.iterrows():
        for day in history_statistic_days:
            if index[0] < day:
                continue
            for feature in features:
                df.set_value(index[0], feature + '_sta_' + str(day), df[index[0] - day: index[0]][feature].mean())
    return df


def _get_history_series(df, window_size):
    """
    构建时间序列历史信息特征（非统计类）
    """
    features = _get_history_features()
    for index, row in df.iterrows():
        if index[0] < window_size:
            continue
        for day in range(window_size):
            for feature in features:
                df.set_value(index[0], feature + '_his_' + str(day + 1), df.iloc[index[0] - day - 1][feature])
    return df


def _get_selected_features():
    # 1.基础特征
    features = _get_common_features()

    updown_features = _get_updown_features()
    history_features = _get_history_features()
    history_statistic_days = _get_history_statistic_days()
    for day in history_statistic_days:
        # 2.历史趋势特征
        features.extend(list(map(lambda x: x + '_trend_' + str(day), [feature for feature in updown_features])))
        # 3.历史统计类特征
        features.extend(list(map(lambda x: x + '_sta_' + str(day), [feature for feature in history_features])))
    # 4.历史序列类特征
    window_size = _get_history_days()
    for day in range(window_size):
        features.extend(list(map(lambda x: x + '_his_' + str(day + 1), [feature for feature in history_features])))
    return features


if __name__ == '__main__':
    pass
