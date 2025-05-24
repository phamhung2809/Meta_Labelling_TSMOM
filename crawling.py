from vnstock import Vnstock
import yfinance as yf
import pandas as pd

def VN_Stock_close_data(start_time,end_time, list_choice = 'VN30', interval = '1D'):
    stock = Vnstock().stock(symbol='ACB', source='VCI')
    stock_list = stock.listing.symbols_by_group(list_choice)

    futures = pd.DataFrame()

    for ma_ck in stock_list:
        try:
            stock = Vnstock().stock(symbol= ma_ck, source='VCI')
            df = stock.quote.history(start= start_time, end= end_time, interval= interval)
            df = df.set_index('time')
            df = pd.DataFrame(df['close'])
            df.columns = [ma_ck]
            df.index = df.index.date
            futures = pd.concat([futures,df],axis = 1, join = 'outer').sort_index()
        except:
            continue

    if interval != '1D':
      futures['Date']= pd.to_datetime(futures.index, format='%Y-%m-%d')
    else:
      futures['Date'] = pd.to_datetime(futures.index, format='%Y-%m-%d %H:%M:%S')
    futures.set_index('Date', inplace=True)

    return futures

def VN_Stock_fully_data(start_time,end_time, list_choice = 'VN30', interval = '1D'):
    stock = Vnstock().stock(symbol='ACB', source='VCI')
    stock_list = stock.listing.symbols_by_group(list_choice)
    data = []

    for ma_ck in stock_list:
        print(ma_ck)
        try:
            stock = Vnstock().stock(symbol= ma_ck, source='VCI')
            df = stock.quote.history(start= start_time, end= end_time, interval= interval)
            df = df.set_index('time')
            df.name = ma_ck
            if interval != '1D':
              df['Date']= pd.to_datetime(df.index,format='%Y-%m-%d')
            else:
              df['Date'] = pd.to_datetime(df.index,format='%Y-%m-%d %H:%M:%S')
            df.set_index('Date', inplace=True)
            data.append(df)
        except:
            continue


    return data

def EU_Stock_data(start_time,end_time, time_range = 'max'):
    """Lấy dữ liệu giá Close của 50 công ty trên sàn Euro_STOXX 50 vào thời gian cho trước"""

    stock_list = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'TSM', 'BRK.A', 'LLY', 'WMT', 'JPM', 'V', 'MA', 'XOM', 'UNH', 'ORCL', 'COST', 'PG', 'HD', 'ACN', 'WFC', 'CSCO', 'NOW', 'AXP', 'BX', 'MCD', 'IBM', 'AZN', 'PEP', 'TMO', 'AMD', 'MS', 'DIS', 'ABT', 'PFE', 'CVX', 'CRM', 'INTC', 'CMCSA', 'LIN', 'NKE', 'T', 'MDT', 'UNP', 'HON', 'PM', 'KO', 'MRK']

    futures = pd.DataFrame()

    # xét từng mã
    for symbol in stock_list:
        try:
            df = yf.Ticker(symbol).history(period = time_range, start = start_time, end = end_time)
            df = pd.DataFrame(df['Close'])
            df.columns = [symbol]
            df.index = df.index.date
            futures = pd.concat([futures,df],axis = 1, join = 'outer').sort_index()
        except:
            continue

    futures['Date'] = pd.to_datetime(futures.index, format='%Y-%m-%d')
    futures.set_index('Date', inplace=True)

    return futures