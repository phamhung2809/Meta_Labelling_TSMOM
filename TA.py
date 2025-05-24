import pandas as pd
import numpy as np


def MA(data, period):
   return data.rolling(period).mean()

def EMA(data, period):
    return pd.Series(data.ewm(ignore_na=False, span=period, adjust=True).mean())

def MACD(data,period_fast = 1,period_slow = 5):
    '''Hàm dùng để tính MACD
       INPUT:
          data(series): Dãy dữ liệu giá của chứng khoán trong 1 khoảng thời gian
          period_fast, period_slow: Chu trình fast,slow; mặc định là (1,5) do số lượng dữ liệu giới hạn
       OUTPUT: MACD(series): Dãy kết quả MACD
    '''
    EMA_fast = EMA(data,period_fast)
    EMA_Slow = EMA(data,period_slow)
    return EMA_fast - EMA_Slow

def RSI(data, period  = 6):
    '''Hàm dùng để tính RSI (đã scaled)
       INPUT:
          data(series): Dãy dữ liệu giá của chứng khoán trong 1 khoảng thời gian
          period: Chu trình RSI; mặc định là 5  do số lượng dữ liệu giới hạn
       OUTPUT: RSI(Series): Dãy kết quả RSI
    '''
    # Tính sự thay đổi hằng ngày
    delta = data.diff()

    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    # Tính trung bình tăng/ trung bình giảm trong chu kì
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Tính rsi
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))) / 100

    return rsi

def Psy_line(data, period = 110):
    better_candle = data.rolling(2).apply(lambda x: ((x.iloc[1] - x.iloc[0])>0).astype(int))
    return better_candle.rolling(period).sum()/period

def ROC(data, period):
    # return data.diff(period)
    return data.pct_change(period)

def Bollinger_Bands(data, period=20, num_std=2):
    '''Tính Bollinger Bands
       OUTPUT: DataFrame với cột 'Middle', 'Upper', 'Lower'
    '''
    middle_band = data.rolling(period).mean()
    std = data.rolling(period).std()

    upper_band = middle_band + (num_std * std)
    lower_band = middle_band - (num_std * std)

    return pd.DataFrame({
        'Middle': middle_band,
        'Upper': upper_band,
        'Lower': lower_band
    })


def Williams_R(data, period=14):
    '''Tính Williams %R
       INPUT: data (DataFrame có cột 'High', 'Low', 'Close')
    '''
    high_max = data['High'].rolling(window=period).max()
    low_min = data['Low'].rolling(window=period).min()
    
    williams_r = -100 * ((high_max - data['Close']) / (high_max - low_min))
    return williams_r


def ATR(data, period=14):
    '''Tính ATR (Average True Range)
       INPUT: data (DataFrame có cột 'High', 'Low', 'Close')
    '''
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def ADX(data, period=14):
    '''Tính ADX (Average Directional Index)
       INPUT: data (DataFrame có cột 'High', 'Low', 'Close')
    '''
    high = data['High']
    low = data['Low']
    close = data['Close']

    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = (-minus_dm).where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def Chaikin_Oscillator(data, short_period=3, long_period=10):
    '''Tính Chaikin Oscillator - Volume Accumulation Oscillator
       INPUT: DataFrame với cột 'High', 'Low', 'Close', 'Volume'
    '''
    adl = ((2 * data['Close'] - data['High'] - data['Low']) / 
           (data['High'] - data['Low']) * data['Volume']).fillna(0).cumsum()
    
    ema_short = adl.ewm(span=short_period, adjust=False).mean()
    ema_long = adl.ewm(span=long_period, adjust=False).mean()
    
    return ema_short - ema_long

