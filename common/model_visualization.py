#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/15 14:23
@Description: TODO
"""
import math

import matplotlib.pyplot as plt


def model_visualization(y_true, y_pred):
    """
    预测结果可视化
    """
    x = range(1, len(y_true) + 1)

    plt.figure(figsize=(10, 6))
    plt.title('copper price forecast model evaluating')
    plt.xlabel('samples')
    plt.ylabel('actual price vs. predict price')
    plt.grid(x)

    plt.plot(x, y_true, '-', label='actual price')
    plt.plot(x, y_pred, '-', label='predict price')

    plt.legend(loc='upper right')

    plt.show()


def model_visulaization_multi_step(y_true, y_pred):
    """
    多步预测的预测结果可视化
    """
    y_true = y_true.T
    y_pred = y_pred.T

    step_len = len(y_true)
    # 计算显示step_len个子图所需的行数和列数
    fig_row_num = int(math.sqrt(step_len))
    fig_col_num = math.ceil(float(step_len) / fig_row_num) if step_len > fig_row_num * fig_row_num else fig_row_num

    x = range(1, len(y_pred[0]) + 1)

    fig, axes = plt.subplots(nrows=fig_row_num, ncols=fig_col_num, sharex=True, sharey=True)
    fig.suptitle('copper price forecast model evaluating - x:samples, y:prices - blue:act, yellow:pre')

    i = 0
    for row in axes:
        for col in row:
            if i < step_len:
                col.set_title("day - " + str(i + 1))
                col.plot(x, y_true[i], '-', label='act')
                col.plot(x, y_pred[i], '-', label='pre')
                # 隐藏legend
                # col.legend(loc='upper right')
                i += 1
    fig.tight_layout()
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
