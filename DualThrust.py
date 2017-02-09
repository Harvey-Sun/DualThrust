# -*- coding:utf-8 -*-

from CloudQuant import MiniSimulator
import numpy as np
import pandas as pd

username = 'Harvey_Sun'
password = 'P948894dgmcsy'
Strategy_Name = 'DualThrust'

INIT_CAP = 100000000
START_DATE = '20130101'
END_DATE = '20161231'
window = 5
k1 = 0.75
k2 = 0.5
Fee_Rate = 0.001
program_path = 'K:/cStrategy/'


def initial(sdk):
    # 准备数据
    sdk.prepareData(['LZ_GPA_QUOTE_THIGH', 'LZ_GPA_QUOTE_TLOW', 'LZ_GPA_QUOTE_TCLOSE',
                     'LZ_GPA_INDEX_CSI500MEMBER', 'LZ_GPA_SLCIND_STOP_FLAG'])
    stock_position = dict()
    sdk.setGlobal('stock_position', stock_position)
    buy_and_hold = []
    buy_and_hold_time = []
    sdk.setGlobal('buy_and_hold', buy_and_hold)
    sdk.setGlobal('buy_and_hold_time', buy_and_hold_time)


def init_per_day(sdk):
    stock_position = sdk.getGlobal('stock_position')
    buy_and_hold = sdk.getGlobal('buy_and_hold')
    buy_and_hold_time = sdk.getGlobal('buy_and_hold_time')
    sdk.clearGlobal()
    sdk.setGlobal('stock_position', stock_position)
    sdk.setGlobal('buy_and_hold', buy_and_hold)
    sdk.setGlobal('buy_and_hold_time', buy_and_hold_time)

    today = sdk.getNowDate()
    sdk.sdklog(today, '========================================日期')
    # 获取当天中证500成分股
    in_zz500 = pd.Series(sdk.getFieldData('LZ_GPA_INDEX_CSI500MEMBER')[-1]) == 1
    stock_list = sdk.getStockList()
    zz500 = list(pd.Series(stock_list)[in_zz500])
    sdk.setGlobal('zz500', zz500)
    # 获取仓位信息
    positions = sdk.getPositions()
    sdk.sdklog(len(positions), '持有股票数量')
    stock_with_position = [i.code for i in positions]
    # 找到中证500外的有仓位的股票
    out_zz500_stock = list(set(stock_with_position) - set(zz500))
    # 以下代码获取当天未停牌未退市的股票，即可交易股票
    # not_stop = pd.isnull(sdk.getFieldData('LZ_GPA_SLCIND_STOP_FLAG')[-1])  # 当日没有停牌的股票
    not_stop = pd.isnull(sdk.getFieldData('LZ_GPA_SLCIND_STOP_FLAG')[-(window + 1):]).all(axis=0)  # 当日和前window日均没有停牌的股票
    zz500_available = list(pd.Series(stock_list)[np.logical_and(in_zz500, not_stop)])
    sdk.setGlobal('zz500_available', zz500_available)
    # 以下代码获取当天被移出中证500的有仓位的股票中可交易的股票
    out_zz500_available = list(set(pd.Series(stock_list)[not_stop]).intersection(set(out_zz500_stock)))
    sdk.setGlobal('out_zz500_available', out_zz500_available)
    # 订阅所有可交易的股票
    stock_available = list(set(zz500_available + out_zz500_available))
    sdk.sdklog(len(stock_available), '订阅股票数量')
    sdk.subscribeQuote(stock_available)
    # 找到所有可交易股票前window日四个价位
    high = pd.DataFrame(sdk.getFieldData('LZ_GPA_QUOTE_THIGH')[-window:], columns=stock_list)[stock_available]
    low = pd.DataFrame(sdk.getFieldData('LZ_GPA_QUOTE_TLOW')[-window:], columns=stock_list)[stock_available]
    close = pd.DataFrame(sdk.getFieldData('LZ_GPA_QUOTE_TCLOSE')[-window:], columns=stock_list)[stock_available]
    HH = high.max(axis=0)
    HC = close.max(axis=0)
    LC = close.min(axis=0)
    LL = low.min(axis=0)
    # 计算range
    s1 = HH - HC
    s2 = LC - LL
    Range = pd.Series(np.where(s1 > s2, s1, s2), index=HH.index)
    # 全局变量
    sdk.setGlobal('Range', Range)
    # 建立一个列表，来记录当天有过交易的股票
    traded_stock = []
    sdk.setGlobal('traded_stock', traded_stock)


def strategy(sdk):
    if (sdk.getNowTime() >= '093000') & (sdk.getNowTime() < '150000'):
        today = sdk.getNowDate()
        # 获取仓位信息及有仓位的股票
        positions = sdk.getPositions()
        position_dict = dict([[i.code, i.optPosition] for i in positions])
        stock_with_position = [i.code for i in positions]
        # number = len(stock_with_position)
        #  找到中证500外的有仓位的股票
        zz500 = sdk.getGlobal('zz500')
        out_zz500_stock = list(set(stock_with_position) - set(zz500))
        out_num = len(out_zz500_stock)
        # 找到目前有仓位且可交易的中证500外的股票
        out_zz500_available = sdk.getGlobal('out_zz500_available')
        out_zz500_tradable = list(set(out_zz500_stock).intersection(set(out_zz500_available)))
        # 获得中证500当日可交易的股票
        zz500_available = sdk.getGlobal('zz500_available')
        # 加载已交易股票
        traded_stock = sdk.getGlobal('traded_stock')
        # 有底仓的股票
        stock_position = sdk.getGlobal('stock_position')
        number = sum(stock_position.values()) / 2  # 计算有多少个全仓股
        # 无仓位股票可用资金
        available_cash = sdk.getAccountInfo().availableCash / (500 - number)
        # 底仓开平记录
        buy_and_hold = sdk.getGlobal('buy_and_hold')
        buy_and_hold_time = sdk.getGlobal('buy_and_hold_time')
        # 开盘时计算上下轨
        if sdk.getNowTime() == '093000':
            stock_available = list(set(zz500_available + out_zz500_available))
            # 获取开盘价
            quotes = sdk.getQuotes(stock_available)
            open_prices = []
            for stock in stock_available:
                open_prices.append(quotes[stock].open)
            Open = pd.Series(open_prices, index=stock_available)
            # 计算上下轨
            Range = sdk.getGlobal('Range')
            up_line = Open + k1 * Range
            down_line = Open - k2 * Range
            sdk.setGlobal('up_line', up_line)
            sdk.setGlobal('down_line', down_line)
            # 建立底仓
            stock_to_build_base = list(set(zz500_available) - set(stock_position.keys()))
            base_hold = []
            date_and_time = []
            for stock in stock_to_build_base:
                price = quotes[stock].current
                volume = 100 * np.floor(available_cash * 0.5 / (100 * price))
                if volume > 0:
                    order = [stock, price, volume, 1]
                    base_hold.append(order)
                    date_and_time.append([today, '093000'])
                    stock_position[stock] = 1
                    traded_stock.append(stock)
            sdk.makeOrders(base_hold)
            sdk.sdklog(len(traded_stock), '=======建立底仓股票数量')
            buy_and_hold += base_hold
            buy_and_hold_time += date_and_time

        # 去除今天已经有交易的股票，获得当下还可交易的股票
        zz500_tradable = list(set(zz500_available) - set(traded_stock))
        # 取得盘口数据
        quotes = sdk.getQuotes(zz500_tradable + out_zz500_tradable)
        # 上下轨
        up_line = sdk.getGlobal('up_line')
        down_line = sdk.getGlobal('down_line')

        # 考虑被移出中证500的那些股票
        sell_orders_out500 = []
        date_and_time = []
        if out_zz500_tradable:
            for stock in out_zz500_tradable:
                position = position_dict[stock]
                current_price = quotes[stock].current
                down = down_line[stock]
                if current_price < down:  # 判断是否卖出
                    order = [stock, current_price, position, -1]
                    sell_orders_out500.append(order)
                    date_and_time.append([today, sdk.getNowTime()])
                    del stock_position[stock]
        sdk.makeOrders(sell_orders_out500)
        buy_and_hold += sell_orders_out500
        buy_and_hold_time += date_and_time
        # 考虑当日中证500可交易的股票
        buy_orders = []
        sell_orders = []
        for stock in zz500_tradable:
            # 如果当时买入股票超过了500-number?
            current_price = quotes[stock].current
            up = up_line[stock]
            down = down_line[stock]
            if (current_price > up) & (stock_position[stock] == 0):
                volume = 100 * np.floor(available_cash / (100 * current_price))
                if volume > 0:
                    order = [stock, current_price, volume, 1]
                    buy_orders.append(order)
                    traded_stock.append(stock)  # 这里有待考虑，下单后不一定会成交,现假设都能成交
                    stock_position[stock] = 2
            if (current_price > up) & (stock_position[stock] == 1):
                volume = 100 * np.floor(available_cash * 0.5 / (100 * current_price))
                if volume > 0:
                    order = [stock, current_price, volume, 1]
                    buy_orders.append(order)
                    traded_stock.append(stock)  # 这里有待考虑，下单后不一定会成交,现假设都能成交
                    stock_position[stock] = 2
            elif (current_price < down) & (stock in stock_with_position):
                volume = position_dict[stock]
                order = [stock, current_price, volume, -1]
                sell_orders.append(order)
                traded_stock.append(stock)
                stock_position[stock] = 0
            else:
                pass
        sdk.makeOrders(sell_orders)
        sdk.makeOrders(buy_orders)
        # 记录下单数据
        if buy_orders or sell_orders or sell_orders_out500:
            sdk.sdklog(sdk.getNowTime(), '=================时间')
            if buy_orders:
                sdk.sdklog('Buy orders')
                sdk.sdklog(np.array(buy_orders))
            if sell_orders:
                sdk.sdklog('Sell orders')
                sdk.sdklog(np.array(sell_orders))
            if sell_orders_out500:
                sdk.sdklog('Sell removed stocks')
                sdk.sdklog(np.array(sell_orders_out500))

        sdk.setGlobal('traded_stock', traded_stock)
        sdk.setGlobal('stock_position', stock_position)
        sdk.setGlobal('buy_and_hold', buy_and_hold)
        sdk.setGlobal('buy_and_hold_time', buy_and_hold_time)
    if (sdk.getNowDate() == '20161230') & (sdk.getNowTime() == '150000'):
        buy_and_hold = sdk.getGlobal('buy_and_hold')
        buy_and_hold_time = sdk.getGlobal('buy_and_hold_time')
        temp = pd.DataFrame(buy_and_hold_time)
        temp = pd.concat([temp, pd.Series(buy_and_hold)], axis=1)
        pd.DataFrame(temp).to_csv('buy_and_hold.csv')


config = {
    'username': username,
    'password': password,
    'initCapital': INIT_CAP,
    'startDate': START_DATE,
    'endDate': END_DATE,
    'strategy': strategy,
    'initial': initial,
    'preparePerDay': init_per_day,
    'feeRate': Fee_Rate,
    'strategyName': Strategy_Name,
    'logfile': '%s.log' % Strategy_Name,
    'rootpath': program_path,
    'executeMode': 'M',
    'feeLimit': 5,
    'cycle': 1,
    'dealByVolume': True,
    'allowForTodayFactors': ['LZ_GPA_INDEX_CSI500MEMBER', 'LZ_GPA_SLCIND_STOP_FLAG']
}

if __name__ == "__main__":
    # 在线运行所需代码
    import os
    config['strategyID'] = os.path.splitext(os.path.split(__file__)[1])[0]
    MiniSimulator(**config).run()
