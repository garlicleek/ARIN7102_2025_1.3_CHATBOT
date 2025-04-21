import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


def split_train(data, test_months=4):
    train_data = data.iloc[:-test_months]
    return data, train_data


def prepare_data(full_data, train_data, lookback=12):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)
    X_train, y_train = [], []
    for i in range(lookback, len(scaled_train)):
        X_train.append(scaled_train[i - lookback:i])
        y_train.append(scaled_train[i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # 测试集序列
    combined_data = scaler.transform(full_data)
    X_test, y_test = [], []
    for i in range(len(train_data), len(combined_data)):
        X_test.append(combined_data[i - lookback:i])
        y_test.append(combined_data[i])
    X_test, y_test = np.array(X_test), np.array(y_test)
    return X_train, y_train, X_test, y_test, scaler


def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mse, mae, mape


# 自定义训练函数
class MetricsLogger(Callback):
    def __init__(self, X_train, y_train, X_test, y_test, scaler):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler
        self.metrics = {
            'train_mse': [], 'train_mae': [], 'train_mape': [],
            'test_mse': [], 'test_mae': [], 'test_mape': []
        }

    def on_epoch_end(self, epoch, logs=None):
        # 预测并计算指标
        y_train_pred = self.model.predict(self.X_train, verbose=0)
        y_train_true= self.scaler.inverse_transform(self.y_train)
        y_train_pred = self.scaler.inverse_transform(y_train_pred)

        y_test_pred = self.model.predict(self.X_test, verbose=0)
        y_test_true = self.scaler.inverse_transform(self.y_test)
        y_test_pred = self.scaler.inverse_transform(y_test_pred)

        # 计算指标
        train_mse, train_mae, train_mape = evaluate_predictions(y_train_true, y_train_pred)
        test_mse, test_mae, test_mape = evaluate_predictions(y_test_true, y_test_pred)
        self.metrics['train_mse'].append(train_mse)
        self.metrics['train_mae'].append(train_mae)
        self.metrics['train_mape'].append(train_mape)
        self.metrics['test_mse'].append(test_mse)
        self.metrics['test_mae'].append(test_mae)
        self.metrics['test_mape'].append(test_mape)

        # 打印当前epoch结果
        print(
            f"Epoch {epoch + 1}: "
            f"train_MSE={self.metrics['train_mse'][-1]:.2f}, "
            f"train_MAE={self.metrics['train_mae'][-1]:.2f}, "
            f"train_MAPE={self.metrics['train_mape'][-1]:.2f}%"
            f"test_MSE={self.metrics['test_mse'][-1]:.2f}, "
            f"test_MAE={self.metrics['test_mae'][-1]:.2f}, "
            f"test_MAPE={self.metrics['test_mape'][-1]:.2f}%"
        )

    def plot_metrics(self):
        plt.figure(figsize=(15, 10))

        # 绘制MSE曲线
        plt.subplot(3, 1, 1)
        plt.plot(self.metrics['train_mse'], label='Train MSE', color='blue')
        plt.plot(self.metrics['test_mse'], label='Test MSE', color='red')
        plt.title('MSE over Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid()

        # 绘制MAE曲线
        plt.subplot(3, 1, 2)
        plt.plot(self.metrics['train_mae'], label='Train MAE', color='blue')
        plt.plot(self.metrics['test_mae'], label='Test MAE', color='red')
        plt.title('MAE over Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid()

        # 绘制MAPE曲线
        plt.subplot(3, 1, 3)
        plt.plot(self.metrics['train_mape'], label='Train MAPE', color='blue')
        plt.plot(self.metrics['test_mape'], label='Test MAPE', color='red')
        plt.title('MAPE over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('MAPE (%)')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()


def rolling_forecast(model, initial_data, scaler, lookback=12, forecast_steps=12):
    forecast = []
    scaled_data = scaler.transform(initial_data)
    current_input = scaled_data[-lookback:].reshape(1, lookback, -1)
    for i in range(forecast_steps):
        next_pred = model.predict(current_input, verbose=0)[0]
        forecast.append(next_pred)
        current_input = np.concatenate([current_input[:, 1:, :], next_pred.reshape(1, 1, -1)], axis=1)
    forecast = scaler.inverse_transform(np.array(forecast))
    return forecast


def append_forecast_to_data(original_df, forecast, forecast_start='2021-01-01'):
    forecast_dates = pd.date_range(
        start=forecast_start,
        periods=len(forecast),
        freq='MS'  # 每月第一天
    )
    forecast_df = pd.DataFrame(
        forecast,
        index=forecast_dates,
        columns=original_df.columns
    )
    full_df = pd.concat([original_df, forecast_df])
    full_df = full_df.reset_index().rename(columns={'index': 'ds'})

    return full_df




root_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(root_path, 'LSTM_data.csv')
df = pd.read_csv(data_path)
df['ds'] = pd.to_datetime(df['ds'])
data = df.set_index('ds')

test_months = 4
epochs = 20
batch_size = 8
lookback = 12
forecast_steps = 12

full_data, train_data = split_train(data, test_months)
# print(data.shape, train_data.shape)
X_train, y_train, X_test, y_test, scaler = prepare_data(full_data, train_data, lookback)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(lookback, full_data.shape[1])),
    LSTM(16),
    Dense(full_data.shape[1])
])
model.compile(optimizer='adam', loss='mse')
plotter = MetricsLogger(X_train, y_train, X_test, y_test, scaler)
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[plotter],
    verbose=1
)
plotter.plot_metrics()

forecast = rolling_forecast(model, full_data, scaler, lookback, forecast_steps)

# print(forecast.shape)
# 最终测试集结果
# final_pred = model.predict(X_test)
# final_pred = scaler.inverse_transform(final_pred)
# final_true = scaler.inverse_transform(y_test)
# 异常值检测
# def has_negative_np(matrix):
#     return np.any(matrix < 0)
#
# matrix = forecast
# print(has_negative_np(matrix))
# # 获取所有负值的索引
# negative_indices = np.where(matrix < 0)
# print("负值坐标：", list(zip(*negative_indices)))
#
# # 统计负值数量
# negative_count = np.sum(matrix < 0)

final_result = append_forecast_to_data(data, forecast, forecast_start='2021-01-01')
print(final_result.shape)
result_file = os.path.join(root_path, "LSTM_result.csv")
final_result.to_csv(result_file, index=False)