import numpy as np
from typing import Dict, Optional, Tuple, Any
from .base import BaseFaultDetector


class BoltLoosenDetector(BaseFaultDetector):
    """
    螺栓松动检测器
    基于峭度（Kurtosis）和峰值因子（Crest Factor）联合判断
    """

    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        super().__init__(name, config)
        # 检测器参数
        self.window_size = self.params.get("window_size", 10)          # 历史基线窗口
        self.trigger_count = self.params.get("trigger_count", 3)       # 连续超阈值次数
        self.kurtosis_threshold = self.params.get("kurtosis_threshold", 6.0)
        self.crest_factor_threshold = self.params.get("crest_factor_threshold", 5.0)
        # 组合逻辑：可选 'and' 或 'or'
        self.combine_logic = self.params.get("combine_logic", "and")   # 默认两个指标都超才算

        # 传感器独立配置（支持每个传感器不同阈值）
        self.sensor_configs = {}
        for sensor_info in config.get("parsed_sensors", []):
            sensor_name = sensor_info["name"]
            self.sensor_configs[sensor_name] = sensor_info

        # 内部状态
        self.fault_counter = {}     # sensor_name -> 连续超阈值次数
        # 可选：记录历史指标用于动态基线，本版本使用固定阈值

    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        kurt = data_packet.get("kurtosis")
        cf = data_packet.get("crest_factor")
        timestamp = data_packet.get("timestamp")

        if kurt is None or cf is None:
            self.mylog.warning(f"传感器 {sensor_name} 数据包缺少峭度或峰值因子")
            return False, None

        # 获取该传感器的自定义阈值（如果配置了）
        sensor_cfg = self.sensor_configs.get(sensor_name, {})
        k_thresh = sensor_cfg.get("kurtosis_threshold", self.kurtosis_threshold)
        c_thresh = sensor_cfg.get("crest_factor_threshold", self.crest_factor_threshold)
        logic = sensor_cfg.get("combine_logic", self.combine_logic)

        # 判断是否超过各自阈值
        k_over = kurt > k_thresh
        c_over = cf > c_thresh

        # 组合判断
        if logic == "and":
            is_over = k_over and c_over
        elif logic == "or":
            is_over = k_over or c_over
        else:
            self.mylog.error(f"无效的组合逻辑: {logic}，使用默认 'and'")
            is_over = k_over and c_over

        # 计数器更新
        cnt = self.fault_counter.get(sensor_name, 0)
        if is_over:
            cnt += 1
        else:
            cnt = 0
        self.fault_counter[sensor_name] = cnt

        is_fault = cnt >= self.trigger_count

        extra_info = {
            "fault_type": self.name,
            "kurtosis": kurt,
            "crest_factor": cf,
            "k_threshold": k_thresh,
            "c_threshold": c_thresh,
            "k_over": k_over,
            "c_over": c_over,
            "combine_logic": logic,
            "counter": cnt,
            "trigger_count": self.trigger_count
        }

        if is_fault:
            self.mylog.info(f"螺栓松动故障触发: sensor={sensor_name}, "
                            f"峭度={kurt:.2f}(>{k_thresh}), 峰值因子={cf:.2f}(>{c_thresh}), "
                            f"连续次数={cnt}")

        return is_fault, extra_info

    def reset(self, sensor_name: Optional[str] = None):
        if sensor_name:
            self.fault_counter.pop(sensor_name, None)
        else:
            self.fault_counter.clear()