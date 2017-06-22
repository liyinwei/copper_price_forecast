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

    plt.plot(x, actual, '-', label='actual price')
    plt.plot(x, predict, '-', label='predict price')

    plt.legend(loc='upper right')

    plt.show()


def plot_loss(loss, val_loss):
    """
    打印每个epochs的loss及val_loss
    """
    x = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.title('loss and val_loss of model')
    plt.xlabel('epochs')
    plt.ylabel('loss and val_loss')
    plt.grid(x)

    plt.plot(x, loss, '-', label='loss')
    plt.plot(x, val_loss, '-', label='val_loss')

    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    pass
