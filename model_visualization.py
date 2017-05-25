#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/15 14:23
@Description: TODO
"""
import matplotlib.pyplot as plt


def model_visualization(actual, predict):
    """
    预测结果可视化
    """
    x = range(1, len(actual) + 1)

    plt.figure(figsize=(10, 6))
    plt.title('copper price forecast model evaluating')
    plt.xlabel('samples')
    plt.ylabel('actual price vs. predict price')
    plt.grid(x)

    plt.plot(x, actual, 'x-', label='actual price')
    plt.plot(x, predict, '+-', label='predict price')

    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    pass
