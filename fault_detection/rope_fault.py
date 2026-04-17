# rope_fault.py
import time
from typing import Dict, Optional, Tuple, Any
from loguru import logger


class RopeFaultDetector:
    """
    钢丝绳/钢带故障检测器（边缘端简化版）
    仅支持：磨损/断丝检测 —— 基于高频 RMS 连续超阈值判断
    """

    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        self.name = name
        self.default_params = config.get('params', {})
        self.sensors_config = config.get('parsed_sensors', [])

        # 默认参数
        self.default_high_rms_threshold = self.default_params.get('high_rms_threshold', 0.15)
        self.default_consecutive_count = self.default_params.get('consecutive_exceed_count', 5)
        self.default_alarm_cooldown = self.default_params.get('alarm_cooldown', 30)

        # 传感器独立配置
        self.sensor_configs = {}
        for sensor_cfg in self.sensors_config:
            sensor_name = sensor_cfg.get('name')
            if sensor_name:
                self.sensor_configs[sensor_name] = sensor_cfg

        # 运行时状态管理
        self._states: Dict[str, _SensorState] = {}
        self._init_states()

        logger.info(f"[{self.name}] 钢丝绳磨损检测器初始化完成（仅高频RMS），管理传感器: {list(self._states.keys())}")

    def _init_states(self):
        for sensor_name, sensor_cfg in self.sensor_configs.items():
            # 合并配置：传感器独立参数优先
            threshold = sensor_cfg.get('high_rms_threshold', self.default_high_rms_threshold)
            count = sensor_cfg.get('consecutive_exceed_count', self.default_consecutive_count)
            cooldown = sensor_cfg.get('alarm_cooldown', self.default_alarm_cooldown)

            self._states[sensor_name] = _SensorState(
                sensor_name, threshold, count, cooldown
            )

    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        """
        处理传感器数据包
        要求 data_packet 包含 'rms_high' 字段（单位：g）
        """
        state = self._states.get(sensor_name)
        if state is None:
            logger.warning(f"[{self.name}] 收到未注册传感器 '{sensor_name}' 的数据，忽略")
            return False, None

        # 仅处理稳态数据（可选）
        if data_packet.get('running_state') != 'steady':
            return False, None

        rms_high = data_packet.get('rms_high')
        if rms_high is None:
            logger.error(f"[{self.name}] 数据包缺少 'rms_high' 字段，传感器: {sensor_name}")
            return False, None

        timestamp = data_packet.get('timestamp', time.time())
        return state.update(rms_high, timestamp)

    def reset(self, sensor_name: Optional[str] = None):
        if sensor_name:
            state = self._states.get(sensor_name)
            if state:
                state.reset()
                logger.info(f"[{self.name}] 传感器 '{sensor_name}' 状态已重置")
        else:
            for state in self._states.values():
                state.reset()
            logger.info(f"[{self.name}] 所有传感器状态已重置")


class _SensorState:
    """单个传感器的磨损检测状态机"""
    def __init__(self, sensor_name: str, threshold: float, consecutive_count: int, cooldown: int):
        self.sensor_name = sensor_name
        self.threshold = threshold
        self.consecutive_count = consecutive_count
        self.cooldown = cooldown

        self.exceed_counter = 0
        self.alarm_triggered = False
        self.last_alarm_time = 0.0

        logger.debug(f"[{sensor_name}] 磨损检测参数: 高频RMS阈值={threshold:.4f}g, "
                     f"连续次数={consecutive_count}, 冷却={cooldown}s")

    def update(self, rms_high: float, timestamp: float) -> Tuple[bool, Optional[Dict]]:
        # 超限判断
        if rms_high > self.threshold:
            self.exceed_counter += 1
            logger.debug(f"[{self.sensor_name}] 高频RMS超限: {rms_high:.4f} > {self.threshold:.4f}, "
                         f"连续 {self.exceed_counter}/{self.consecutive_count}")
        else:
            if self.exceed_counter > 0:
                logger.debug(f"[{self.sensor_name}] 高频RMS回落，计数器清零")
            self.exceed_counter = 0
            if self.alarm_triggered:
                self.alarm_triggered = False
                logger.info(f"[{self.sensor_name}] 磨损报警解除")

        # 报警触发
        alarm_info = None
        if (self.exceed_counter >= self.consecutive_count and
            not self.alarm_triggered and
            not self._in_cooldown(timestamp)):

            self.alarm_triggered = True
            self.last_alarm_time = timestamp
            alarm_info = {
                'fault_type': 'rope_wear',
                'sensor': self.sensor_name,
                'rms_high': rms_high,
                'threshold': self.threshold,
                'exceed_count': self.exceed_counter,
                'message': (f"钢丝绳磨损/断丝预警: {self.sensor_name} "
                            f"高频RMS连续{self.consecutive_count}次超过{self.threshold:.4f}g"),
                'timestamp': timestamp
            }
            logger.warning(alarm_info['message'])

        return self.alarm_triggered, alarm_info

    def _in_cooldown(self, current_time: float) -> bool:
        if self.cooldown <= 0:
            return False
        return (current_time - self.last_alarm_time) < self.cooldown

    def reset(self):
        self.exceed_counter = 0
        self.alarm_triggered = False
        self.last_alarm_time = 0.0