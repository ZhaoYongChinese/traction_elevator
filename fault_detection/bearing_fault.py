import numpy as np
import json
import os
import sys
from typing import Dict, Optional, Tuple, List, Any
from .base import BaseFaultDetector

# 假设 real_path 已定义在工具模块中，此处直接引用
sys.path.append(os.path.join(sys.path[0], 'package'))
from utils import real_path   # 根据实际调整

class BearingFaultDetector(BaseFaultDetector):
    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict):
        super().__init__(name, config)
        # 轴承参数从全局配置获取（也可放在 params 中）
        bearing_args = global_config["bearing"]
        self.window_size = self.params.get("window_size", global_config["FAULT_WINDOW"])
        self.expand_ratio = self.params.get("expand_ratio", 8)
        self.fault_trigger_count = self.params.get("trigger_count", global_config["FAULT_TRIGGER_COUNT"])
        
        self.rms_history = {}
        self.fault_counter = {}
        
        # 计算轴承特征频率
        n = bearing_args["n"]
        d = bearing_args["d"]
        D = bearing_args["D"]
        beta = bearing_args["beta"]
        fr = bearing_args["rpm"] / 60.0
        
        f_inner = 0.5 * n * fr * (1 + d / D * np.cos(beta))
        f_outer = 0.5 * n * fr * (1 - d / D * np.cos(beta))
        f_cage = 0.5 * fr * (1 - d / D * np.cos(beta))
        f_ball = (D / (2 * d)) * fr * (1 - ((d / D) * np.cos(beta)) ** 2)
        self.location_dict = {"inner": f_inner, "outer": f_outer, "cage": f_cage, "ball": f_ball}
        
        # 加载历史 RMS 基线（用于计算阈值）
        try:
            with open(real_path("rms.json"), 'r', encoding='utf-8') as f:
                self.rms_history = json.load(f)
            self.mylog.info(f"Loaded RMS history: {self.rms_history}")
        except Exception as e:
            self.mylog.error(f"Failed to load rms.json: {e}")
    
    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        rms_value = data_packet["rms_value"]
        fft_all = data_packet.get("fft_all")
        timestamp = data_packet["timestamp"]
        
        history = self.rms_history.get(sensor_name, [])
        if len(history) < self.window_size:
            return False, None
        
        max_val = np.max(history)
        threshold = max_val * self.expand_ratio
        over_thresh = rms_value > threshold
        
        cnt = self.fault_counter.get(sensor_name, 0)
        if over_thresh:
            cnt += 1
        else:
            cnt = 0
        self.fault_counter[sensor_name] = cnt
        
        is_fault = cnt >= self.fault_trigger_count
        extra_info = {
            "fault_type": self.name,
            "rms_value": rms_value,
            "threshold": threshold,
            "ratio": rms_value / threshold if threshold != 0 else 0,
            "counter": cnt,
            "trigger_count": self.fault_trigger_count
        }
        
        # 二级故障判断（频谱特征频率）
        if is_fault and fft_all:
            second_result = self._check_second_fault(fft_all["fft"], fft_all["index"])
            if second_result:
                extra_info["second_fault"] = second_result
        
        # 更新历史
        history.append(rms_value)
        if len(history) > self.window_size:
            history.pop(0)
        self.rms_history[sensor_name] = history
        
        return is_fault, extra_info
    
    def _check_second_fault(self, spectrum, freqs, snr_threshold=3, window=2):
        results = {}
        for key, f in self.location_dict.items():
            idx = np.argmin(np.abs(freqs - f))
            amp = spectrum[idx]
            start = max(0, idx - window)
            end = min(len(spectrum), idx + window + 1)
            local_region = np.delete(spectrum[start:end], idx - start)
            local_mean = np.mean(local_region) if len(local_region) > 0 else 1e-6
            if amp > local_mean * snr_threshold:
                results[key] = {"freq": f, "amp": amp, "local_noise": local_mean, "SNR": amp / (local_mean + 1e-12)}
        return results if results else None
    
    def reset(self, sensor_name: Optional[str] = None):
        if sensor_name:
            self.rms_history.pop(sensor_name, None)
            self.fault_counter.pop(sensor_name, None)
        else:
            self.rms_history.clear()
            self.fault_counter.clear()