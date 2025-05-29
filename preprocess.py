from TA import *
import copy


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

def feature_engineering(data, period = 20, day_barrier = 5, pct_barrier = 0.05):
  '''Hàm dùng để tạo feature từ dữ liệu đã được chuyển về daily
      INPUT: data_list(list): Một list bao gồm các dataframe đã được chuyển về daily
        * Lưu ý: Nếu chỉ sử dụng 1 công ty, chỉ cần truyền vào 1 dataframe dưới dạng list: (VD: [df])
      OUTPUT: feature_data(list): Danh sách các dataframe sau khi tạo feature
  '''

  # Tạo 1 bản copy của data
  temp = copy.deepcopy(data)
  temp = temp[temp['close'] != 0]
  temp['Symbol'] = data.name
  # # Tính MA
  for x in [20,60,252]:
    feature = 'RSI' + str(x)
    temp[feature] = RSI(temp['close'],x)
    feature = 'PSY' + str(x)
    temp[feature] = Psy_line(temp['close'],x)

  for x in [1,3,5,10,15,20,40,60]:
    # feature = 'P' + str(x)
    # temp[feature] = temp['close'].shift(x)
    feature = 'ROC' + str(x)
    temp[feature] = ROC(temp['close'],x)

  temp['MACD_1_5'] = MACD(temp['close'],1,5)
  temp['MACD_5_20'] = MACD(temp['close'],5,20)
  temp['MACD_20_60'] = MACD(temp['close'],20,60)

  temp['Volume'] = temp['volume']

  temp['VWAP'] = (((temp['high'] + temp['low'] + temp['close']) / 3)* temp['Volume']).cumsum() / temp['Volume'].cumsum()

  temp['signal_momentum'] = temp['close'].rolling(period).apply(lambda x: x.iloc[period - 1] / x.iloc[0] - 1)
  # temp['signal_momentum'] = [1 if x > momentum_threshold else -1 if x < -momentum_threshold else 0 for x in temp['momentum']]
  # temp['signal_momentum'] = [1 if x > momentum_threshold else 0 for x in temp['momentum']]


  temp['Future_result'] = temp['close'].rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]).shift(-1)

  temp['good_signal'] = triple_barier_labels(temp['close'], day_barrier, pct_barrier)

  # temp['good_signal'] = ((temp["Future_result"] * (temp["signal_momentum"])) > 0).astype(int)

  # temp['good_signal']= ((temp["Future_result"] * np.sign(temp['signal_momentum'])) > 0).astype(int)


  temp.drop(columns = ['open','high','low','volume','close','Future_result','Volume'], inplace = True)

  # Xóa những cột xuất hiện NA (Xảy ra với các hàng đầu của dữ liệu khi không có hơn 5 ngày để quan sát RSI)
  temp.dropna(axis=0, how="any", inplace = True)


  # Đặt tên data để dễ phân biệt
  temp.name = data.name + "_feature"


  return temp
