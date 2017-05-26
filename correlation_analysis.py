#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/25 14:35
@Description: 铜价和PCB价格相关性分析
"""
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

from data_loading import read_co_price, read_pcb_price


def data_visualization(co_price, pcb_price):
    """
    原始数据可视化
    """
    x_co_values = co_price.index
    y_co_values = co_price.price / 100

    x_pcb_values = pcb_price.index
    y_pcb_values = pcb_price.price

    plt.figure(figsize=(10, 6))
    plt.title('history copper price(100rmb/t) vs. pcb price(rmb/sq.m.)')
    plt.xlabel('date')
    plt.ylabel('history price')

    plt.plot(x_co_values, y_co_values, '-', label='co price')
    plt.plot(x_pcb_values, y_pcb_values, '-', label='pcb price')

    plt.legend(loc='upper right')

    plt.show()


def co_price_pre_process(co_price):
    """
    为便于跟PCB价格进行对比，将非交易日的铜价设置为前后两个交易日收盘价的均值
    """
    min_date = min(co_price.index)
    max_date = max(co_price.index)
    date_range = pd.date_range(min_date, max_date)
    # 构造一个用于保存每天价格数据的DataFrame
    df = pd.DataFrame(np.full((len(date_range), 1), np.nan), index=date_range, columns=['price'])
    df.update(co_price)
    # 通过线性插值填补缺失值
    return df.interpolate()


def pcb_price_pre_process(pcb_price):
    """
    提取pcb每天的价格 
    """
    # 获取报价日期范围
    min_date = pcb_price.valid_date[0]
    max_date = pcb_price.invalid_date[len(pcb_price) - 1]
    date_range = pd.date_range(min_date, max_date)
    # 构造一个用于保存每天价格数据的DataFrame
    df = pd.DataFrame(np.zeros(len(date_range)), index=date_range, columns=['price'])
    # 上一条记录的valid_date
    pre_start_date = None
    # 上一条记录的invalid_date
    pre_end_date = None
    # 上一条记录的price
    pre_price = None
    for index, row in pcb_price.iterrows():
        if index == 0:
            # 若是第一条，则仅保存
            pre_start_date = row.valid_date
            pre_end_date = row.invalid_date
            pre_price = row.price
            continue
        current_start_date = row.valid_date
        current_max_date = max(pre_end_date, current_start_date)
        # 将上一条的valid_date至max(上一条的max, 当前条的valid_date)的价格设置为上一条记录的price
        df.set_value(pd.date_range(pre_start_date, current_max_date if current_max_date < max_date else max_date),
                     'price', pre_price)
        # 更新当前记录至上一条记录的变量中
        pre_start_date = current_start_date
        pre_end_date = row.invalid_date
        pre_price = row.price
    df.to_excel('price.xlsx')
    return df


def filter_data_by_date(df, start_date, end_date):
    """
    根据开始日期和截止日期对dataframe进行切片
    """
    if start_date not in df.index:
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        start_date = start_date - datetime.timedelta(days=1)
    if end_date not in df.index:
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        end_date = end_date + datetime.timedelta(days=1)
    return df[start_date:end_date]


def cor_analysis(co_price, pcb_price):
    """
    铜价和PCB价格相关性分析 
    """
    cor_draw(co_price, pcb_price)
    print(pearsonr(co_price.values, pcb_price.values))


def cor_draw(co_price, pcb_price):
    """
    横坐标：铜价
    纵坐标：PCB价格
    """
    plt.figure(figsize=(10, 6))
    plt.title('the correlation of copper price & pcb price')
    plt.xlabel('copper price')
    plt.ylabel('pcb price')

    plt.plot(co_price, pcb_price, '-', label='correlation')

    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    # 可视化数据的日期范围
    # START_DATE = '2002-01-07'
    START_DATE = '2014-10-23'
    END_DATA = '2017-05-24'
    # 读取铜价数据
    co_price_data = read_co_price()
    # 对铜价数据进行缺失值填充
    co_price_data = co_price_pre_process(co_price_data)
    # 过滤出可视化日期范围内的数据
    co_price_data = filter_data_by_date(co_price_data, START_DATE, END_DATA)
    # 读取PCB报价数据
    pcb_price_data = read_pcb_price()
    # 对PCB数据进行处理
    pcb_price_data = pcb_price_pre_process(pcb_price_data)
    # 过滤出可视化日期范围内的数据
    pcb_price_data = filter_data_by_date(pcb_price_data, START_DATE, END_DATA)
    # 将两者进行相关性分析
    cor_analysis(co_price_data/100, pcb_price_data)
    # 可视化铜价历史数据及PCB价格历史数据
    data_visualization(co_price_data, pcb_price_data)
