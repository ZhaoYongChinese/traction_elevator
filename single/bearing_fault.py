# bearing_fault.py
import json
import os
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from loguru import logger


class BearingFaultDetector:
    """
    轴承故障检测器（独立版）
    一级筛查：实时 RMS 与历史窗口最大值比较（超阈值倍数）
    二级诊断：频谱特征频率（内圈、外圈、保持架、滚动体）信噪比检测
    """

    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        self.name = name
        self.default_params = config.get('params', {})
        self.sensors_config = config.get('parsed_sensors', [])

        # 轴承参数（可从 global_config 或 params 中获取）
        if global_config:
            bearing_args = global_config.get("bearing", {})
        else:
            bearing_args = self.default_params.get("bearing", {})

        # 全局配置参数
        if global_config:
            self.fault_window = global_config.get("FAULT_WINDOW", 10)
            self.fault_trigger_count = global_config.get("FAULT_TRIGGER_COUNT", 3)
        else:
            self.fault_window = self.default_params.get("window_size", 10)
            self.fault_trigger_count = self.default_params.get("trigger_count", 3)

        self.window_size = self.default_params.get("window_size", self.fault_window)
        self.expand_ratio = self.default_params.get("expand_ratio", 8)
        self.fault_trigger_count = self.default_params.get("trigger_count", self.fault_trigger_count)

        # 历史文件路径（优先从配置读取，否则使用默认文件名）
        self.rms_history_file = self.default_params.get("rms_history_file", "bearing_rms_history.json")

        # 传感器独立配置
        self.sensor_configs = {}
        for sensor_cfg in self.sensors_config:
            sensor_name = sensor_cfg.get('name')
            if sensor_name:
                self.sensor_configs[sensor_name] = sensor_cfg

        # 运行时状态
        self.rms_history: Dict[str, List[float]] = {}
        self.fault_counter: Dict[str, int] = {}

        # 计算轴承特征频率
        if bearing_args:
            n = bearing_args.get("n", 9)
            d = bearing_args.get("d", 12.7)
            D = bearing_args.get("D", 65.0)
            beta = bearing_args.get("beta", 0)
            rpm = bearing_args.get("rpm", 1450)
            fr = rpm / 60.0

            f_inner = 0.5 * n * fr * (1 + d / D * np.cos(np.radians(beta)))
            f_outer = 0.5 * n * fr * (1 - d / D * np.cos(np.radians(beta)))
            f_cage = 0.5 * fr * (1 - d / D * np.cos(np.radians(beta)))
            f_ball = (D / (2 * d)) * fr * (1 - ((d / D) * np.cos(np.radians(beta))) ** 2)
            self.location_dict = {"inner": f_inner, "outer": f_outer, "cage": f_cage, "ball": f_ball}
        else:
            self.location_dict = {}
            logger.warning(f"[{self.name}] 未提供轴承参数，二级频域诊断将不可用")

        # 加载历史 RMS 基线
        self._load_rms_history()

        logger.info(f"[{self.name}] 轴承故障检测器初始化完成，管理传感器: {list(self.sensor_configs.keys())}")

    def _load_rms_history(self):
        """从 JSON 文件加载历史 RMS 基线"""
        if os.path.exists(self.rms_history_file):
            try:
                with open(self.rms_history_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # 确保加载的数据格式正确
                    if isinstance(loaded, dict):
                        self.rms_history = loaded
                    else:
                        self.rms_history = {}
                logger.info(f"[{self.name}] 加载 RMS 历史文件: {self.rms_history_file}")
            except Exception as e:
                logger.error(f"[{self.name}] 加载 RMS 历史文件失败: {e}")
                self.rms_history = {}
        else:
            logger.info(f"[{self.name}] 未找到历史文件 {self.rms_history_file}，将从头开始收集数据")

    def _save_rms_history(self):
        """保存当前 RMS 历史到 JSON 文件"""
        try:
            with open(self.rms_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.rms_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[{self.name}] 保存 RMS 历史文件失败: {e}")

    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        rms_value = data_packet.get("rms_value")
        fft_all = data_packet.get("fft_all")
        timestamp = data_packet.get("timestamp")

        if rms_value is None:
            logger.warning(f"[{self.name}] 传感器 {sensor_name} 数据包缺少 rms_value")
            return False, None

        # 获取该传感器历史
        history = self.rms_history.get(sensor_name, [])
        if len(history) < self.window_size:
            # 数据不足，只记录不诊断
            history.append(rms_value)
            if len(history) > self.window_size:
                history.pop(0)
            self.rms_history[sensor_name] = history
            self._save_rms_history()
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
            "expand_ratio": self.expand_ratio,
            "ratio": rms_value / threshold if threshold != 0 else 0,
            "counter": cnt,
            "trigger_count": self.fault_trigger_count
        }

        # 二级频域诊断
        if is_fault and fft_all and self.location_dict:
            try:
                spectrum = np.array(fft_all["fft"])
                freqs = np.array(fft_all["index"])
                second_result = self._check_second_fault(spectrum, freqs)
                if second_result:
                    extra_info["second_fault"] = second_result
            except Exception as e:
                logger.error(f"[{self.name}] 二级频域诊断失败: {e}")

        # 更新历史并保存
        history.append(rms_value)
        if len(history) > self.window_size:
            history.pop(0)
        self.rms_history[sensor_name] = history
        self._save_rms_history()

        return is_fault, extra_info

    def _check_second_fault(self, spectrum: np.ndarray, freqs: np.ndarray,
                            snr_threshold: float = 3.0, window: int = 2) -> Optional[Dict]:
        """检测特征频率处的信噪比是否超阈值"""
        results = {}
        for key, f in self.location_dict.items():
            idx = np.argmin(np.abs(freqs - f))
            amp = spectrum[idx]
            start = max(0, idx - window)
            end = min(len(spectrum), idx + window + 1)
            local_region = np.delete(spectrum[start:end], idx - start)
            local_mean = np.mean(local_region) if len(local_region) > 0 else 1e-6
            if amp > local_mean * snr_threshold:
                results[key] = {
                    "freq": f,
                    "amp": float(amp),
                    "local_noise": float(local_mean),
                    "SNR": float(amp / (local_mean + 1e-12))
                }
        return results if results else None

    def reset(self, sensor_name: Optional[str] = None):
        if sensor_name:
            self.rms_history.pop(sensor_name, None)
            self.fault_counter.pop(sensor_name, None)
        else:
            self.rms_history.clear()
            self.fault_counter.clear()
        self._save_rms_history()
        logger.info(f"[{self.name}] 检测器状态已重置")