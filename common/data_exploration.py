#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/12 9:48
@Description: 铜价历史价格可视化
"""

import matplotlib.pyplot as plt

from common.data_loading import read_co_data


def data_visualization(co_price):
    """
    原始数据可视化
    """
    x_co_values = co_price.price_date
    y_co_values = co_price.close_price_i

    plt.figure(figsize=(10, 6))
    plt.title('history copper price(rmb/t)')
    plt.xlabel('')
    plt.ylabel('history price')

    plt.plot(x_co_values, y_co_values, '-', label='co price')

    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    # 读取铜价数据
    co_price_data = read_co_data()
    # 可视化铜价历史数据及PCB价格历史数据
    data_visualization(co_price_data)
