from TA import *
import copy
import dateutil.parser
from changepoint import *

from changepoynt.algorithms.bocpd import BOCPD  # import the scoring algorithm
from changepoynt.visualization.score_plotting import plot_data_and_score  # import a visualization function

def Volatility_scale(data, ignore_na=False, adjust = True, com = 60, min_periods=0):
    day_vol = data.ewm(ignore_na=ignore_na,
                          adjust=adjust,
                          com=com,
                          min_periods=min_periods).std(bias=False)
    
    vol = day_vol * np.sqrt(252)  # scale lại theo 252 ngày active trading

    return vol


def triple_barier_labels(data,day_barrier, pct_barrier):
  label =  copy.deepcopy(data) * 0
  for i in range (day_barrier, 0, -1):
    temp = data.pct_change(i)
    flag = 0
    for x,v in enumerate(temp.index):
      if np.isnan(temp.loc[v]): continue
      if temp.loc[v] >= pct_barrier: 
        label.loc[v] = i
        flag +=1
      elif temp.loc[v] <= -pct_barrier: 
        label.loc[v] = -i
        flag +=1
    # print(f"{i}: {flag}")
  return label


def feature_engineering(data, period = 30, day_barrier = 20, pct_barrier = 0.03, changepoint = 'BOCD'):

  # Tạo 1 bản copy của data
  temp = copy.deepcopy(data)

  temp = temp[temp['close'] != 0]

  for x in [5,10,20,40]:
    feature = 'RSI' + str(x)
    temp[feature] = RSI(temp['close'],x)

  for x,y in [[1,5],[5,10],[20,40]]:
    feature = 'MACD_' + str(x) +'_' + str(y)
    temp[feature] = MACD(temp['close'], x, y)

  temp['VWAP'] = (((temp['high'] + temp['low'] + temp['close']) / 3)* temp['volume']).cumsum() / temp['volume'].cumsum()

  if changepoint == 'BOCD':
    detector = BOCPD(run_length = period)
    temp['changepoint_bocd'] = detector.transform(temp['close'])

  elif changepoint == 'Gauss':
    cpd_df = run_CPD(
    time_series_data=pd.DataFrame(temp['close']),
    lookback_window_length=period,
    start_date = temp.index[0],
    end_date = temp.index[-1],
    use_kM_hyp_to_initialize_kC=True
  )

    temp = pd.concat([temp, cpd_df], axis=1)

  temp['signal_momentum'] = temp['close'].pct_change(period)

  temp['good_signal'] = (((triple_barier_labels(temp['close'], day_barrier, pct_barrier)>0) * ((temp['signal_momentum']> 0))) > 0).astype(int)

  temp['Close'] = temp['close']
  temp['vol'] = Volatility_scale(temp['close'].pct_change(), ignore_na=True, adjust=True, com=60, min_periods=0)

  temp.drop(columns = ['open','high','low','volume','close'], inplace = True)
  temp.dropna(axis=0, how="any", inplace = True)

  return temp


def split_train_test(data, train_size=0.7, validation_size = 0.1, lstm = False, sequence_length = 40):
    data = data.sort_index()

    split_idx_1 = int(len(data) * train_size)

    split_idx_2 = int(len(data) * (train_size+validation_size)) - int(sequence_length *30 if lstm == True else 0 )

    train, val, test = data.loc[data.index <= dateutil.parser.parse(str(data.iloc[[split_idx_1]].index[0]))], data.loc[(data.index > dateutil.parser.parse(str(data.iloc[[split_idx_1]].index[0]))) & (data.index <= dateutil.parser.parse(str(data.iloc[[split_idx_2]].index[0])))], data.loc[data.index > dateutil.parser.parse(str(data.iloc[[split_idx_2]].index[0]))] 

    return train, val, test


def prepare_X_y(df, keep = []):

    columns = df.columns.tolist()
    remove_list = ['Symbol', 'Close', 'good_signal','vol']
    remove_list = list(set(remove_list) - set(keep))
    columns = list(set(columns) - set(remove_list))
    X = df[columns]
    info = df[['Symbol', 'Close','vol']]
    y = df['good_signal']

    return X, y , info