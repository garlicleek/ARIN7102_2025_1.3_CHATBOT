import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 加载数据
root_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(root_path, 'test_data', 'Abatatriptan.csv')
data = pd.read_csv(data_path)
# data = pd.read_csv("/Users/zirunzhao/Documents/ARIN_7102/ARIN7102_project/Zyvance.csv")

# 将年月合并为日期
# data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'].astype(str))
data = data.sort_values('ds')

# 只保留日期和销量列
data = data.set_index('ds')

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 定义创建数据集函数
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# 设置时间步长（用多少个月预测下一个月）
look_back = 12
X, y = create_dataset(scaled_data, look_back)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 调整输入格式为 [样本数, 时间步长, 特征数]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = models.Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train, y_train, 
                    epochs=100, 
                    batch_size=1, 
                    verbose=1, 
                    validation_data=(X_test, y_test))

# 预测训练集和测试集
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反标准化
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# 计算RMSE
from sklearn.metrics import mean_squared_error
train_score = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
test_score = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
print(f'Train Score: {train_score:.2f} RMSE')
print(f'Test Score: {test_score:.2f} RMSE')

original_data = data.copy()
scaled_data = scaler.fit_transform(data[['y']])


def improved_visualization(original_data, scaled_data, train_predict, test_predict, look_back, scaler):
    """
    original_data: 原始DataFrame（包含日期索引）
    scaled_data: 标准化后的数据（NumPy数组）
    train_predict: 训练集预测结果
    test_predict: 测试集预测结果
    look_back: 时间步长
    scaler: 标准化器对象
    """
    # 获取原始日期索引
    dates = original_data.index
    
    # 创建预测结果的日期范围
    train_dates = dates[look_back:look_back+len(train_predict)]
    test_dates = dates[look_back+len(train_predict):look_back+len(train_predict)+len(test_predict)]
    
    # 绘制图形
    plt.figure(figsize=(14, 7))
    plt.plot(dates, original_data['y'], label='Actual Sales', color='blue')
    plt.plot(train_dates, train_predict, label='Training Predictions', color='green')
    plt.plot(test_dates, test_predict, label='Test Predictions', color='red')
    
    plt.title('Zyvance Sales Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Sales Quantity')
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用改进的可视化（传入原始data而不是scaled_data）
improved_visualization(original_data, scaled_data, train_predict, test_predict, look_back, scaler)

'''
# 创建用于绘图的序列
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict

test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(look_back*2)+1:len(scaled_data)-1, :] = test_predict

# 绘制原始数据和预测数据
plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(scaled_data), label='Actual Data')
plt.plot(train_predict_plot, label='Training Prediction')
plt.plot(test_predict_plot, label='Testing Prediction')
plt.legend()
plt.title('Zyvance Sales Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('Quantity')
plt.show()
'''
# 使用最后look_back个月的数据预测下一个月
last_data = scaled_data[-look_back:]
last_data = np.reshape(last_data, (1, look_back, 1))
next_month_pred = model.predict(last_data)
next_month_pred = scaler.inverse_transform(next_month_pred)
print(f'Predicted sales for next month: {next_month_pred[0][0]:.0f}')

# 可以递归预测更多未来月份
def predict_future(n_months):
    future_predictions = []
    current_batch = scaled_data[-look_back:]
    
    for i in range(n_months):
        current_pred = model.predict(current_batch.reshape(1, look_back, 1))
        future_predictions.append(current_pred[0][0])
        current_batch = np.append(current_batch[1:], current_pred)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

future_6_months = predict_future(6)
print('Predicted sales for next 6 months:')
print(future_6_months)