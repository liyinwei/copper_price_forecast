#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: liyinwei
@E-mail: coridc@foxmail.com
@Time: 2017/5/25 16:53
@Description: 常量类
"""
import sys


class Const:
    """
    自定义常量：（1）命名全部大写；（2）值不可修改
    """
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        def __init__(self):
            pass
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__.keys():
            raise self.ConstError('Can not change const.{0}'.format(name))
        if not name.isupper():
            raise self.ConstCaseError(
                'const name {0} is not all uppercase.'.format(name))
        self.__dict__[name] = value

    # 数据库连接host
    HOST = 'mysqlhost'
    # 数据库访问用户名
    USER = 'cvte'
    # 数据库访问密码
    PASSWORD = 'cvte@cvte'
    # 数据库名称
    DATABASE = 'dataset'

    # 读取期铜交易数据SQL语句
    CO_DATA_SQL = """
        SELECT
            -- 主键
            id,
            -- 交易日期
            price_date,
            -- 交割月份
            delivery_month,
            -- 前结算
            pre_settlement_price_i,
            -- 今开盘
            open_price_i,
            -- 最高价-主力
            highest_price_i,
            -- 最低价-主力
            lowest_price_i,
            -- 收盘价-主力
            close_price_i,
            -- 结算参考价-主力
            settlement_price_i,
            -- 涨跌1
            zd1_chg,
            -- 涨跌2
            zd2_chg,
            -- 成交手-主力
            volume_i,
            -- 持仓手
            open_interest,
            -- 变化
            open_interest_chg,
            -- 综合指数(o_curproduct)
            -- 最高价-综合
            highest_price_p,
            -- 最低价-综合
            lowest_price_p,
            -- 加权平均价-综合
            avg_price_p,
            -- 成交手-综合
            volume_p,
            -- 成交额(亿元)
            turn_over,
            -- 年成交手(万手)
            year_volume,
            -- 年成交额(亿元)
            year_turn_over,
            -- 金属指数(o_curmetalindex)
            -- 最新价
            last_price,
            -- 今开盘价-金属
            open_price_m,
            -- 今收盘价-金属
            close_price_m,
            -- 昨收盘价-金属
            pre_close_price_m,
            -- 涨跌-金属(页面未显示)
            updown,
            -- 涨跌1-金属
            updown1,
            -- 涨跌2-金属
            updown2,
            -- 最高价-金属
            highest_price_m,
            -- 最低价-金属
            lowest_price_m,
            -- 加权平均价-金属
            avg_price_m,
            -- 结算参考价-金属
            settlement_price_m,
            -- others 13
            -- 年份
            o_year,
            -- 月份
            o_month,
            -- 日
            o_day,
            -- 星期
            o_weekday,
            -- 年期序号
            o_year_num
        FROM
            (
                SELECT
                    *
                FROM
                    shfe_daily
                WHERE
                    product_id != '总计'
                AND product_name != '小计'
                ORDER BY
                    volume_i
            ) AS a
        WHERE
            a.product_name = '铜'
        AND a.price_date > '2015-01-05'
        GROUP BY
            a.price_date,
            a.product_id
        ORDER BY
            a.price_date;
    """

    # 读取期铜价格SQL语句
    CO_PRICE_SQL = """
        SELECT
            price_date,
            avg_price_p AS price
        FROM
            shfe_daily
        GROUP BY
            price_date
        ORDER BY
            price_date;
    """

    # 读取PCB价格SQL语句
    PCB_PRICE_SQL = """
        
    """

sys.modules[__name__] = Const()
