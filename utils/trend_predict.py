import numpy as np
from sklearn.linear_model import LinearRegression


class TrendPredict:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.rms_history = {}

    def update(self, sensor_name, rms_value):
        """提取历史数据并进行线性预测"""
        if sensor_name not in self.rms_history:
            self.rms_history[sensor_name] = []
        self.rms_history[sensor_name].append(rms_value)

        if len(self.rms_history[sensor_name]) < self.window_size:
            return None
        else:
            X = np.arange(self.window_size).reshape(-1, 1)
            y = np.array(self.rms_history[sensor_name])
            model = LinearRegression().fit(X, y)
            next_x = np.array([[self.window_size]])
            predict_val = model.predict(next_x)[0]
            self.rms_history[sensor_name].pop(0)
            return predict_val