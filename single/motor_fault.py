# motor_fault.py
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from scipy.signal import find_peaks
from loguru import logger


class MotorFaultDetector:
    """
    电机故障检测器：定子偏心 / 转子不对中
    一级筛查：实时 RMS 与历史最大值比较（超阈值倍数）
    二级诊断：FFT 频谱提取基频及倍频幅值比
    """

    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        self.name = name
        self.params = config.get('params', {})
        
        # 检测器全局参数
        self.window_size = self.params.get("window_size", 10)
        self.default_expand_ratio = self.params.get("expand_ratio", 8)
        self.fault_trigger_count = self.params.get("trigger_count", 3)
        self.sample_rate = self.params.get("sample_rate", global_config.get("PROC_SAMPLING_RATE", 8000) if global_config else 8000)

        # 电机诊断阈值
        self.ratio2_misalign = self.params.get("ratio2_misalign", 0.6)
        self.ratio3_misalign = self.params.get("ratio3_misalign", 0.3)
        self.ratio2_eccentric = self.params.get("ratio2_eccentric", 0.3)
        self.ratio3_eccentric = self.params.get("ratio3_eccentric", 0.15)

        # 传感器独立配置（从 parsed_sensors 中提取）
        self.sensor_configs = {}
        for sensor_info in config.get("parsed_sensors", []):
            sensor_name = sensor_info.get("name")
            if sensor_name:
                self.sensor_configs[sensor_name] = sensor_info

        # 内部状态
        self.rms_history = {}       # sensor_name -> list of RMS values
        self.fault_counter = {}     # sensor_name -> counter

        logger.info(f"[{self.name}] 电机故障检测器初始化完成，管理传感器: {list(self.sensor_configs.keys())}")

    def _get_features_from_fft(self, spectrum: np.ndarray, freqs: np.ndarray) -> Optional[Dict]:
        """从 FFT 幅度谱和频率轴提取基频及倍频特征"""
        peaks, _ = find_peaks(spectrum, height=0.1 * np.max(spectrum))
        if len(peaks) == 0:
            return None

        sorted_idx = sorted(peaks, key=lambda x: spectrum[x], reverse=True)
        f1 = freqs[sorted_idx[0]]
        amp1 = spectrum[sorted_idx[0]]

        def get_amp(target_freq, tolerance=1.0):
            idx = np.argmin(np.abs(freqs - target_freq))
            if abs(freqs[idx] - target_freq) <= tolerance:
                return spectrum[idx]
            else:
                return 0.0

        amp2 = get_amp(2 * f1)
        amp3 = get_amp(3 * f1)
        amp4 = get_amp(4 * f1)

        return {
            "f1": f1,
            "amp1": amp1,
            "amp2": amp2,
            "amp3": amp3,
            "amp4": amp4,
            "ratio2": amp2 / amp1 if amp1 != 0 else 0,
            "ratio3": amp3 / amp1 if amp1 != 0 else 0,
            "ratio4": amp4 / amp1 if amp1 != 0 else 0
        }

    def _diagnose(self, features: Dict) -> Tuple[str, float]:
        """根据倍频比值诊断故障类型"""
        r2 = features["ratio2"]
        r3 = features["ratio3"]

        if r2 >= self.ratio2_misalign and r3 >= self.ratio3_misalign:
            fault_type = "转子不对中"
            confidence = min(1.0, (r2 + r3) / 1.5)
        elif r2 >= self.ratio2_eccentric and r3 >= self.ratio3_eccentric:
            fault_type = "定子偏心"
            confidence = min(1.0, (r2 + r3) / 1.0)
        else:
            if r2 >= r3:
                fault_type = "转子不对中"
                confidence = min(0.5, r2)
            else:
                fault_type = "定子偏心"
                confidence = min(0.5, r3)

        return fault_type, confidence

    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        rms_value = data_packet.get("rms_value")
        fft_all = data_packet.get("fft_all")
        timestamp = data_packet.get("timestamp")

        if rms_value is None:
            logger.warning(f"[{self.name}] 传感器 {sensor_name} 数据包缺少 rms_value")
            return False, None

        sensor_cfg = self.sensor_configs.get(sensor_name, {})
        expand_ratio = sensor_cfg.get("custom_threshold", self.default_expand_ratio)

        history = self.rms_history.get(sensor_name, [])
        history.append(rms_value)
        if len(history) > self.window_size:
            history.pop(0)
        self.rms_history[sensor_name] = history

        if len(history) < self.window_size:
            return False, None

        max_val = np.max(history)
        threshold = max_val * expand_ratio
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
            "expand_ratio": expand_ratio,
            "ratio": rms_value / threshold if threshold != 0 else 0,
            "counter": cnt,
            "trigger_count": self.fault_trigger_count
        }

        if is_fault and fft_all is not None:
            try:
                spectrum = np.array(fft_all["fft"])
                freqs = np.array(fft_all["index"])
                features = self._get_features_from_fft(spectrum, freqs)
                if features is not None:
                    motor_fault, confidence = self._diagnose(features)
                    extra_info["motor_fault_type"] = motor_fault
                    extra_info["confidence"] = confidence
                    extra_info["freq_features"] = {
                        "f1": features["f1"],
                        "amp1": features["amp1"],
                        "ratio2": features["ratio2"],
                        "ratio3": features["ratio3"]
                    }
                    logger.info(f"[{self.name}] 电机故障诊断: {motor_fault} (置信度 {confidence:.2f})")
                else:
                    logger.warning(f"[{self.name}] 无法从频谱提取特征")
            except Exception as e:
                logger.error(f"[{self.name}] 电机故障精细诊断失败: {e}")

        return is_fault, extra_info

    def reset(self, sensor_name: Optional[str] = None):
        if sensor_name:
            self.rms_history.pop(sensor_name, None)
            self.fault_counter.pop(sensor_name, None)
        else:
            self.rms_history.clear()
            self.fault_counter.clear()