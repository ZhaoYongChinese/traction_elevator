import numpy as np

class SteadyStateDetector:
    def __init__(self, window_size=50, threshold_ratio=0.15):
        """
        window_size: 滑动窗口大小（点数），用于计算局部RMS
        threshold_ratio: RMS变异系数阈值，低于此值视为平稳
        """
        self.window_size = window_size
        self.threshold_ratio = threshold_ratio
        self.rms_buffer = []
    
    def is_steady(self, signal_chunk):
        """
        输入一段信号（如Z轴），返回是否为稳态
        策略：将信号分段，计算每段RMS，若RMS波动小于阈值则判定为稳态
        """
        if len(signal_chunk) < self.window_size * 2:
            return False
        
        # 分段计算RMS
        n_segments = len(signal_chunk) // self.window_size
        rms_list = []
        for i in range(n_segments):
            seg = signal_chunk[i*self.window_size : (i+1)*self.window_size]
            rms_list.append(np.sqrt(np.mean(seg**2)))
        
        if len(rms_list) < 3:
            return False
        
        # 计算RMS的变异系数（标准差/均值）
        mean_rms = np.mean(rms_list)
        if mean_rms < 1e-6:
            return False  # 信号过弱，可能停机
        cv = np.std(rms_list) / mean_rms
        
        return cv < self.threshold_ratio