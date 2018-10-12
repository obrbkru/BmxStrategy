import json
import random
import unittest
import math
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from sqlite3 import Error
import time

stopStategy = 0
takeStategy = 0
positionSize = 10000
takerFee = -0.025
makerFee = 0.075

class HystoryDbWrapper():
    def __init__(self, path):
        try:
            self.conn = sqlite3.connect(path)
            print(sqlite3.version)
        except Error as e:
            print(e)
            self.conn.close()

    def get_records(self, table_name, field='*'):
        request = 'SELECT ' + field + ' FROM ' + table_name + ';'
        cur = self.conn.cursor()
        res = cur.execute(request)
        return cur.fetchall()


class Position:
    def __init__(self, strategy, openPrice , side, openTime):
        self.openPrice = openPrice
        self.side = side
        self.openTime = openTime
        self.take = 0
        self.stop = 0
        self.closePrice = 0.0
        self.closeTime = 0
        self.strategy = strategy

    def Close(self, closePrice, closeTime):
        self.closePrice = closePrice
        self.closeTime = closeTime

    def GetStatus(self):
        if self.closeTime == 0.0 and self.openTime == 0.0:
            return 'NotStarted'
        elif self.closeTime == 0.0:
            return 'InProcess'
        else:
            if self.closePrice == self.take:
                return 'FinishedWithProffit'
            elif self.closePrice == self.GetStopSize():
                return 'FinishedWithDamages'
            else:
                return 'FinishedBreakeven'

    def IsClosed(self):
        return not self.closeTime == 0.0

    def Erase(self):
        self.openPrice = 0
        self.side = 0
        self.openTime = 0
        self.take = 0
        self.stop = 0

    def IsEmpty(self):
        return self.openPrice == 0

def wma(prices, period):
    """
    Weighted Moving Average (WMA) is a type of moving average that assigns a
    higher weighting to recent price data.

    WMA = (P1 + 2 P2 + 3 P3 + ... + n Pn) / K
    where K = (1+2+...+n) = n(n+1)/2 and Pn is the most recent price after the
    1st WMA we can use another formula
    WMAn = WMAn-1 + w.(Pn - SMA(prices, n-1))
    where w = 2 / (n + 1)

    http://www.csidata.com/?page_id=797

    http://www.financialwebring.org/gummy-stuff/MA-stuff.htm

    http://www.investopedia.com/terms/l/linearlyweightedmovingaverage.asp

    http://fxtrade.oanda.com/learn/forex-indicators/weighted-moving-average

    Input:
      prices ndarray
      period int > 1 and < len(prices)

    Output:
      wmas ndarray

    Test:

    >>> import numpy as np
    >>> import technical_indicators as tai
    >>> prices = np.array([77, 79, 79, 81, 83, 49, 55])
    >>> period = 5
    >>> print(tai.wma(prices, period))
    [ 80.73333333  70.46666667  64.06666667]
    """

    num_prices = len(prices)

    if num_prices < period:
        # show error message
        raise SystemExit('Error: num_prices < period')

    wma_range = num_prices - period + 1

    wmas = np.zeros(wma_range)

    k = (period * (period + 1)) / 2.0

    # only required for the commented code below
    #w = 2 / float(period + 1)

    for idx in range(wma_range):
        for period_num in range(period):
            weight = period_num + 1
            wmas[idx] += prices[idx + period_num] * weight
        wmas[idx] /= k

    # this is the code for the second formula, but I think the first is simpler
    # to understand
    #for idx in range(wma_range):
        #if idx == 0:
            #for period_num in range(period):
                #weight = period_num + 1
                #wmas[idx] += prices[idx + period_num] * weight
            #wmas[idx] /= k

        #else:
            #wmas[idx] = wmas[idx - 1] + w * \
                #(prices[idx + period - 1] - \
                 #sma(prices[idx - 1:idx + period - 1], period))

    return wmas

def rsi(prices, period=14):
    num_prices = len(prices)

    if num_prices < period:
        # show error message
        raise SystemExit('Error: num_prices < period')

    # this could be named gains/losses to save time/memory in the future
    changes = prices[1:] - prices[:-1]
    #num_changes = len(changes)

    rsi_range = num_prices - period

    rsis = np.zeros(rsi_range)

    gains = np.array(changes)
    # assign 0 to all negative values
    masked_gains = gains < 0
    gains[masked_gains] = 0

    losses = np.array(changes)
    # assign 0 to all positive values
    masked_losses = losses > 0
    losses[masked_losses] = 0
    # convert all negatives into positives
    losses *= -1

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsis[0] = 100
    else:
        rs = avg_gain / avg_loss
        rsis[0] = 100 - (100 / (1 + rs))

    for idx in range(1, rsi_range):
        avg_gain = ((avg_gain * (period - 1) + gains[idx + (period - 1)]) /
                    period)
        avg_loss = ((avg_loss * (period - 1) + losses[idx + (period - 1)]) /
                    period)

        if avg_loss == 0:
            rsis[idx] = 100
        else:
            rs = avg_gain / avg_loss
            rsis[idx] = 100 - (100 / (1 + rs))

    return rsis


def GetTakerComission():
    return positionSize * takerFee / 100
def GetMakerComission():
    return positionSize * makerFee / 100

class PredictedPosition:
    def __init__(self, stock, last_price):
        self.side = ''
        self.open = 0
        self.stop = 0
        self.take = 0

        prev_candle = stock.get_hour_candle(-1)
        candle = stock.get_hour_candle()
        ema50 = stock.get_ema_val_by_period(50)
        ema_fast_period = 10
        ema_fast = stock.get_ema_val_by_period(ema_fast_period)
        wma80 = stock.get_wma_val_by_period(80)
        upTail = abs(candle.high - max(candle.open, candle.close))
        downTail = abs(candle.low - min(candle.open, candle.close))
        candleBody = abs(candle.open - candle.close)

        is_red = candle.open > candle.close
        self.stopStategy = 0.5
        self.takeStategy = 0.8
        self.rejectStategy = 0.3

        '''
        if candleBody > candle.close / 320 and candleBody < candle.close / 115 and idRed:
            self.side = 'Sell'
            self.open = last_price + last_price / 170
            self.strategy = 1
            self.stopStategy = 0.3
            self.takeStategy = 0.6

        elif candleBody > candle.close / 320 and candleBody < candle.close / 115 and not idRed:
            self.side = 'Buy'
            self.open = last_price - last_price / 170
            self.strategy = 2
            self.stopStategy = 0.3
            self.takeStategy = 0.6

elif downTail > candle.close / 140:
            self.side = 'Sell'
            self.open = last_price + last_price / 200
            self.strategy = 3
            self.stopStategy = 0.26
            self.takeStategy = 0.60

        elif upTail > last_price / 140:
            self.side = 'Buy'
            self.open = last_price - last_price / 200
            self.strategy = 4
            self.stopStategy = 0.26
            self.takeStategy = 0.60'''

        '''
        if downTail > candle.close * 0.45 / 100 and candleBody < downTail / 2 and upTail < downTail / 5:
            self.side = 'Buy'
            self.open = last_price - last_price * 0.1 / 100
            self.strategy = 1
            self.stopStategy = 0.9
            self.takeStategy = 1.0

        elif upTail > candle.close * 0.45 / 100 and candleBody < upTail / 2 and downTail < upTail / 5:
            self.side = 'Sell'
            self.open = last_price + last_price * 0.1 / 100
            self.strategy = 2
            self.stopStategy = 0.9
            self.takeStategy = 1.0
         '''
        last_price_change = abs(stock.get_hour_candle(-5).open - candle.close)

        sell = True
        if last_price > ema50:
            sell = False
        elif last_price < ema50:
            sell = True
        else:
            return

        shift = 5

        if ema_fast > wma80 and stock.get_ema_val_by_period(ema_fast_period, -1) < stock.get_wma_val_by_period(80, -1):
            self.side = 'Buy'
            self.strategy = 5
            self.open = last_price - shift
            print('ema_fast = %s wma80 = %s prev_ema_fast = %s prevWma80 = %s close = %s' % (ema_fast, wma80, stock.get_ema_val_by_period(ema_fast_period, -1), stock.get_wma_val_by_period(80, -1), candle.close))
        elif ema_fast < wma80 and stock.get_ema_val_by_period(ema_fast_period, -1) > stock.get_wma_val_by_period(80, -1):
            self.side = 'Sell'
            self.strategy = 6
            self.open = last_price + shift
            if int(wma80) == 6584:
                t = 0
            print('ema_fast = %s wma80 = %s prev_ema_fast = %s prevWma80 = %s close = %s' % (ema_fast, wma80, stock.get_ema_val_by_period(ema_fast, -1), stock.get_wma_val_by_period(80, -1), candle.close))
        else:
            return
        '''
        k = abs(last_price - ema50)
        shift = 50#abs(last_price*0.02 -
        if last_price > ema50:
            self.side = 'Buy'
            self.strategy = 5
            self.open = ema50 - shift
        elif last_price < ema50:
            self.side = 'Sell'
            self.strategy = 6
            self.open = ema50 + shift
        else:
            return'''
        #if self.side == 'Buy':
        #    self.open = last_price - last_price * 0.3 / 100
        #else :
        #    self.open = last_price + last_price * 0.3 / 100

        self.take = self.GetTakeSize()
        self.stop = self.GetStopSize()
        self.reject = self.GetRejectSize(last_price)
        self.check_values(last_price)

    def check_values(self, last_price):
        if self.side != '':
            assert (self.open)
            assert (self.take)
            assert (self.stop)
            assert (self.reject)
        if self.side == 'Buy':
            assert (self.open < last_price)
            assert (self.stop < self.open)
            assert (self.take > self.open)
            assert (self.reject > last_price)
        elif self.side == 'Sell':
            assert (self.open > last_price)
            assert (self.stop > self.open)
            assert (self.take < self.open)
            assert (self.reject < last_price)

    def AddSign(self, shift):
        if self.side == 'Sell':
            return (-1 * shift)
        elif self.side == 'Buy':
            return shift

    def GetStopCondition(self):
        shift = self.open * self.takeStategy / 100
        return self.open + self.AddSign(shift)

    def GetTakeSize(self):
        shift = self.open * self.takeStategy / 100
        return self.open + self.AddSign(shift)

    def GetRejectSize(self, last_price):
        shift = last_price * self.rejectStategy / 100
        return last_price + self.AddSign(shift)

    def GetStopSize(self):
        shift = self.open * self.stopStategy / 100
        return self.open - self.AddSign(shift)

class Candle:
    def __init__(self, open, high, low, close, volume, openTime, closeTime, length = 1):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.openTime = openTime
        self.closeTime = closeTime
        self.length = length

    def IsRed(self):
        return self.close < self.open

def Filter(candle):
    global fullRsi
    global currentCandle
    global turnOnFilter
    global wma60trend
    global ema5
    global ema20
    global status
    global g_candles
    direction = ''

    prevCandle = g_candles[currentCandle - 10]
    candleBody = abs(candle.open - candle.close)
    upTail = abs(candle.high - max(candle.open, candle.close))
    downTail = abs(candle.low - min(candle.open, candle.close))

    if candle.IsRed():
        return 'short'
    else:
        return 'long'
    '''if downTail > candle.close / 140:
        return 'short'
    elif upTail > candle.close / 140:
        return 'long'

    if candleBody > candle.close / 320 and candleBody < candle.close / 115 and candle.IsRed():
        return 'short'
    elif candleBody > candle.close / 320 and candleBody < candle.close / 115 and not candle.IsRed():
        return 'long'''

    '''if candleBody > candle.close / 40 :
        direction = ''
    elif fullRsi[currentCandle] > 72.0:
        direction = 'short'
    elif fullRsi[currentCandle] < 44.0:
        direction = 'long'
    elif wma60trend == 'down':
        direction = 'long'
    elif wma60trend == 'up':
        direction = 'short'

    if direction == 'long' and candleBody > downTail:
        direction = ''

    if direction == 'long' and not candle.IsRed():
        direction = ''
    elif direction == 'short' and candle.IsRed():
        direction = ''
    if g_status == 'FinishedWithDamages':
        if g_failedDirection == 'long' and direction == 'short':
            direction = ''
        elif g_failedDirection == 'short' and direction == 'long':
            direction = '''

    return direction

def GetTriggeredEvent(currentPrice, newPrice, position):
    assert (position.take != 0)
    assert (position.stop != 0)
    assert (newPrice != 0)
    assert(currentPrice != 0)
    isMoveUp = newPrice > currentPrice
    if position.side == 'Sell':
        if isMoveUp :
            if position.stop <= newPrice:
                return 'stop'
            else:
                return ''
        else:
            if position.take >= newPrice:
                return 'take'
            else:
                return ''
    else:
        if isMoveUp :
            if position.take <= newPrice:
                return 'take'
            else:
                return ''
        else:
            if position.stop >= newPrice:
                return 'stop'
            else:
                return ''

class Bitmex:
    def __init__(self):
        db = HystoryDbWrapper('F:/trading/hystory.db')
        records = db.get_records('bitmex_candles_1m')
        #records = db.get_records('binance_candles_1m')
        self.ema_dict = {}
        self.wma_dict = {}
        self.minute_candles = []
        self.hour_candles = []
        self.hour_candles_close_price = []
        self.current_minute_candle = 0
        self.day_candles = []

        day_candle = None
        hour_candle = None
        hours_counter = 0
        counter = 0

        for res in records:
            minute_candle = Candle(open=float(res[2]), high=float(res[3]), low=float(res[4]), close=float(res[5]),
                            volume=float(res[6]), openTime=datetime.fromtimestamp(res[1]),
                            closeTime=datetime.fromtimestamp(res[7]))
            self.minute_candles.append(minute_candle)
            if counter == 0:
                if minute_candle.openTime.minute != 0:
                    t = 0
                hour_candle = Candle(open=minute_candle.open, high=0, low=0, close=0,
                                     volume=0, openTime=minute_candle.openTime, closeTime=0)
                if minute_candle.openTime == datetime.fromtimestamp(1538377200):
                    t = 0
                if hours_counter == 0:
                    day_candle = Candle(open=minute_candle.open, high=0, low=0, close=0,
                                         volume=0, openTime=minute_candle.openTime, closeTime=0)
            hour_candle.volume += minute_candle.volume
            day_candle.volume += minute_candle.volume
            hour_candle.high, hour_candle.low = self.get_high_and_low(minute_candle, hour_candle.high, hour_candle.low)
            day_candle.high, day_candle.low = self.get_high_and_low(minute_candle, day_candle.high, day_candle.low)

            counter += 1
            if counter == 60 :
                counter = 0
                hour_candle.close = minute_candle.close
                hour_candle.closeTime = minute_candle.closeTime
                self.hour_candles.append(hour_candle)
                self.hour_candles_close_price.append(hour_candle.close)
                hours_counter += 1
                hour_candle = None
                if hours_counter == 24:
                    hours_counter = 0
                    day_candle.close = minute_candle.close
                    day_candle.closeTime = minute_candle.closeTime
                    self.day_candles.append(day_candle)
                    day_candle = None

    def get_high_and_low(self, candle, high, low):
        if low == 0 or candle.low < low:
            low = candle.low
        if candle.high > high:
            high = candle.high
        return high, low

    def get_ema_val_by_period(self, period, num = 0):
        if num > 0:
            return None #unexpected number of candle, expected negative value

        if not period in self.ema_dict:
            pricesDataFrame = pd.DataFrame({'Close': self.hour_candles_close_price})
            self.ema_dict[period] = EMA(pricesDataFrame, period=period)['EMA']

        current_hour_candle = int(self.current_minute_candle/60)
        if current_hour_candle == 0 or current_hour_candle + num < 0:
            return None

        return self.ema_dict[period][current_hour_candle - 1 + num]

    def get_wma_val_by_period(self, period, num = 0):
        if num > 0:
            return None #unexpected number of candle, expected negative value

        if not period in self.wma_dict:
            #pricesDataFrame = pd.DataFrame({'Close': self.hour_candles_close_price})
            #self.wma_dict[period] = pricesDataFrame.rolling(period).mean()['Close']
            self.wma_dict[period] = [0]*80
            self.wma_dict[period].extend(wma(self.hour_candles_close_price, period))

        current_hour_candle = int(self.current_minute_candle/60)
        if current_hour_candle == 0 or current_hour_candle + num < 0:
            return None

        return self.wma_dict[period][current_hour_candle - 1 + num]

    def get_next_minute_candle(self):
        if self.current_minute_candle >= len(self.minute_candles):
            return None
        candle = self.minute_candles[self.current_minute_candle]
        self.current_minute_candle += 1
        return candle

    def get_hour_candle(self, num = 0):
        if num > 0:
            return None #unexpected number of candle, expected negative value

        current_hour_candle = int(self.current_minute_candle/60)
        assert current_hour_candle <= len(self.hour_candles)
        if current_hour_candle == 0 or current_hour_candle + num < 0:
            return None

        candle = self.hour_candles[current_hour_candle - 1 + num]
        return candle

    def get_day_candle(self, num = 0):
        if num > 0:
            return None #unexpected number of candle, expected negative value

        current_day_candle = int(self.current_minute_candle/60/24)
        assert current_day_candle <= len(self.day_candles)
        if current_day_candle == 0 or current_day_candle + num < 0:
            return None

        candle = self.day_candles[current_day_candle - 1 + num]
        return candle

class Trader:
    def __init__(self):
        self.trade_balance = 0
        self.prev_candle_1h = None
        self.position = None
        self.balance = 0
        self.unsuccessTradesAmount = 0
        self.successTradesAmount = 0

    def stop_triggered(self, position, candle):
        global positionSize

        self.balance -= positionSize * abs(position.stop - position.openPrice) / position.openPrice + GetMakerComission()
        self.unsuccessTradesAmount += 1
        position.Close(position.stop, candle.closeTime)
        print('%s failed, balance = %s, side = %s, open = %s, stop = %s, strategy = %s' % (position.openTime.strftime("%Y-%m-%d %H:%M:%S"), self.balance, position.side, position.openPrice, position.stop, position.strategy))

    def take_triggered(self, position, candle):
        global positionSize

        self.balance += positionSize * abs(position.openPrice - position.take) / position.openPrice - GetTakerComission()
        self.successTradesAmount += 1
        position.Close(position.take, candle.closeTime)
        print('%s success, balance = %s, side = %s, open = %s, stop = %s, strategy = %s' % (position.openTime.strftime("%Y-%m-%d %H:%M:%S"), self.balance, position.side, position.openPrice, position.stop, position.strategy))

    def handle_event(self, event, position, candle):
        if event == 'stop':
            self.stop_triggered(position, candle)
            return True
        elif event == 'take':
            self.take_triggered(position, candle)
            return True
        return False

    def simple_strategy_process(self, candle, position):
        if candle.IsRed():
            if not self.handle_event(GetTriggeredEvent(candle.open, candle.high, position), position, candle):
                self.handle_event(GetTriggeredEvent(candle.high, candle.low, position), position, candle)
        else:
            if not self.handle_event(GetTriggeredEvent(candle.open, candle.low, position), position, candle):
                self.handle_event(GetTriggeredEvent(candle.low, candle.high, position), position, candle)
        return

    def trade(self):
        stock = Bitmex()

        candle = stock.get_next_minute_candle()
        predictedPos = None
        while candle:
            if candle.openTime.year == 2018 and candle.openTime.month > 4:
                if not self.position:
                    if predictedPos:
                        #if candle.openTime.minute == 0:
                        #    predictedPos = PredictedPosition(stock, candle.open)

                        if (predictedPos.side == 'Sell' and predictedPos.reject >= candle.open) or (
                            predictedPos.side == 'Buy' and predictedPos.reject <= candle.open):
                            predictedPos = None

                        elif ((predictedPos.side == 'Sell' and candle.high >= predictedPos.open) or
                            (predictedPos.side == 'Buy' and candle.low <= predictedPos.open)):
                            self.position = Position(predictedPos.strategy, predictedPos.open, predictedPos.side, candle.openTime)
                            self.position.stop = predictedPos.stop
                            self.position.take = predictedPos.take
                            self.balance -= GetTakerComission()
                    else:
                        if candle.openTime.minute == 0:
                            tmp = PredictedPosition(stock, candle.open)
                            if tmp.open:
                                predictedPos = tmp

                if self.position:
                    self.simple_strategy_process(candle, self.position)
                    if self.position.IsClosed():
                        self.position = None
                        predictedPos = None
            candle = stock.get_next_minute_candle()


def RSI(df, period=14, column="Close"):
    # wilder's RSI

    delta = df[column].diff()
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0

    rUp = up.ewm(com=period - 1, adjust=False).mean()
    rDown = down.ewm(com=period - 1, adjust=False).mean().abs()

    rsi = 100 - 100 / (1 + rUp / rDown)

    return df.join(rsi.to_frame('RSI'))

def SMA(df, column="Close", period=20):

    sma = df[column].rolling(window=period, min_periods=period - 1).mean()
    return df.join(sma.to_frame('SMA'))



def EMA(df, column="Close", period=20):

    ema = df[column].ewm(span=period, min_periods=period - 1).mean()
    return df.join(ema.to_frame('EMA'))

#https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py

def CheckVolumesAndPriceChangesPerTime():
    ticks_data = open('C:/Users/obrbkru/Documents/projects/BBot/binance_BTCUSDT_1h.json').read()
    json_obj = json.loads(ticks_data)
    prices = {}
    volumes = {}
    for i in range(24):
        prices[i] = 0.0
        volumes[i] = 0.0
    fallingTime = 0
    risingTime = 0
    for res in json_obj:
        openTime = datetime.utcfromtimestamp(float(res[0]) / 1000.0)
        closeTime = datetime.utcfromtimestamp(float(res[6]) / 1000.0)
        candle = Candle(open=float(res[1]), high=float(res[2]), low=float(res[3]), close=float(res[4]),
                        volume=float(res[5]), openTime=openTime,
                        closeTime=closeTime)
        if candle.openTime.year == 2018 and candle.openTime.month == 4 :
            if candle.IsRed():
                fallingTime += 1
                prices[openTime.hour] -= candle.open - candle.close
            else:
                prices[openTime.hour] += candle.close - candle.open
                risingTime += 1
            volumes[openTime.hour] += float(res[5])

    for i in range(24):
        print('Time: %s, volume: %s, price change: %s' % (i, volumes[i], prices[i]))
    print('fallingTime: %s, risingTime: %s' % (fallingTime, risingTime))


if __name__ == '__main__':
    #try:
        '''#CheckVolumesAndPriceChangesPerTime()
        #unittest.main()
        turnOnFilter = True
        #ticks_data = open('C:/Users/obrbkru/Documents/projects/BBot/binance_BTCUSDT_1h.json').read()
        #json_obj = json.loads(ticks_data)


        prices = []
        localTrendChanges = []
        totalUnsuccessTrades = 0
        totalSuccessTrades = 0
        curentTrend = 'up'
        totalPrice = 0
        totalVolume = 0


        #oneDollarPerVolume = totalVolume / totalPrice
        #pricesDataFrame = pd.DataFrame({'Close': prices})
        #wma60 = pricesDataFrame.rolling(80).mean()['Close']
        #ema5 = EMA(pricesDataFrame, period=5)['EMA']
        #ema20 = EMA(pricesDataFrame, period=20)['EMA']
        #rsiPeriod = 28
        #fullRsi = RSI(pricesDataFrame, rsiPeriod)['RSI']
        profitPercentage = 0
        candle = None
        balance = 0
        candlesCounter = 0
        for candle in g_candles:
            openTime = candle.openTime
            closeTime = candle.closeTime

            if openTime.hour == 0 and openTime.minute == 0:
                profitPercentage = 0
                if not successTradesAmount == 0:
                    profitPercentage = 100 / ((unsuccessTradesAmount + successTradesAmount) / successTradesAmount)
                print('%s balance = %s, profit percentage = %s, trades count = %s, trend = %s' %
                      (closeTime, balance, profitPercentage, unsuccessTradesAmount + successTradesAmount, wma60trend))
                totalBalance += balance
                balance = 0
                totalUnsuccessTrades += unsuccessTradesAmount
                totalSuccessTrades += successTradesAmount
                unsuccessTradesAmount = 0
                successTradesAmount = 0

            candlesCounter += 1
            position = PredictedPosition(candle.open)
            SimpleStrategy(0.3, 0.58, candle)
            currentCandle += 1
        
        print('%s balance = %s, profit percentage = %s, trades count = %s, trend = %s' % (
        candle.closeTime, balance, profitPercentage, unsuccessTradesAmount + successTradesAmount, wma60trend))'''
        trater = Trader()
        trater.trade()
        print('totalBalance = %s totalSuccessTradesAmount=%s totalUnsuccessTrades=%s ratio=%s' %
              (trater.balance, trater.successTradesAmount, trater.unsuccessTradesAmount, trater.unsuccessTradesAmount / trater.successTradesAmount))

    #except Exception as e:
    #    print(e)