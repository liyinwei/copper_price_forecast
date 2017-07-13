#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/12 9:54
@Description: 铜价和PCB价格相关性分析
"""

import mysql.connector
import pandas as pd

from common import const


def read_data_from_mysql(sql):
    """
    根据sql语句查询数据，并以pandas.DataFrame格式返回
    """
    conn = None
    try:
        conn = mysql.connector.connect(host=const.HOST, user=const.USER, password=const.PASSWORD, database=const.DATABASE,
                                       use_unicode=True, charset='utf8')
        df = pd.read_sql(sql, conn)
        return df
    except Exception as e:
        print(e)
    finally:
        conn.close()


def read_co_data():
    """
    获取沪期铜主力历史交易数据
    """
    print("start loading data...")
    sql = const.CO_DATA_SQL
    df = read_data_from_mysql(sql)
    tuples = list(zip(*[range(len(df)), df.price_date]))
    # 添加数据索引
    index = pd.MultiIndex.from_tuples(tuples, names=['id', 'date'])
    df.index = index
    print("loading data finished.")
    return df


def read_co_data_rnn():
    """
    获取沪期铜主力历史交易数据
    """
    print("start loading data...")
    sql = const.CO_PRICE_SQL_RNN
    df = read_data_from_mysql(sql)
    # tuples = list(zip(*[range(len(df)), df.price_date]))
    # # 添加数据索引
    # index = pd.MultiIndex.from_tuples(tuples, names=['id', 'date'])
    # df.index = index
    df.index = pd.DatetimeIndex(df.price_date)
    print("loading data finished.")
    return df


def read_co_price():
    """
    获取沪期铜主力历史价格
    """
    print("start loading data...")
    sql = const.CO_PRICE_SQL
    df = read_data_from_mysql(sql)
    df.index = pd.DatetimeIndex(df.price_date)
    print("loading data finished.")
    return df.drop(['price_date'], axis=1)


def read_pcb_price():
    """
    获取PCB历史价格数据 
    """
    sql = const.PCB_PRICE_SQL
    df = read_data_from_mysql(sql)
    return df


if __name__ == '__main__':
    pass
