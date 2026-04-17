# bolt_loosen.py
import time
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
from collections import deque

class BoltLoosenDetector:
    """
    螺栓松动检测器（基于RMS阈值 + 连续超限确认 + 报警冷却）
    管理多个传感器，每个传感器可独立配置。
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Args:
            name: 检测器名称（如 'bolt_loosen'）
            config: 包含 'params' 和 'parsed_sensors' 的配置字典
        """
        self.name = name
        self.default_params = config.get('params', {})
        self.sensors_config = config.get('parsed_sensors', [])
        
        # 内部状态：sensor_name -> SensorState 实例
        self._states: Dict[str, _SensorState] = {}
        self._init_sensors()
        
        logger.info(f"[{self.name}] 螺栓松动检测器初始化完成，管理传感器: {list(self._states.keys())}")

    def _init_sensors(self):
        """为每个传感器创建状态对象，并合并配置"""
        for sensor_cfg in self.sensors_config:
            sensor_name = sensor_cfg.get('name')
            if not sensor_name:
                logger.error(f"[{self.name}] 传感器配置缺少 'name' 字段，跳过")
                continue
            
            # 合并默认参数与传感器独立参数
            merged_cfg = {**self.default_params, **sensor_cfg}
            self._states[sensor_name] = _SensorState(sensor_name, merged_cfg)

    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        处理指定传感器的数据包，返回 (是否报警, 报警信息字典)

        Args:
            sensor_name: 传感器标识
            data_packet: 包含 'rms_g' 和可选 'timestamp' 的字典

        Returns:
            (is_fault, extra_info)
            extra_info 格式与之前兼容，例如:
            {
                'fault_type': 'bolt_loosen',
                'sensor': sensor_name,
                'rms': float,
                'threshold': float,
                'exceed_count': int,
                'message': str,
                'timestamp': float
            }
        """
        state = self._states.get(sensor_name)
        if state is None:
            logger.warning(f"[{self.name}] 收到未注册传感器 '{sensor_name}' 的数据，忽略")
            return False, None

        rms_value = data_packet.get('rms_g')
        if rms_value is None:
            logger.error(f"[{self.name}] 数据包缺少 'rms_g' 字段，无法处理传感器 '{sensor_name}'")
            return False, None

        timestamp = data_packet.get('timestamp', time.time())
        return state.update(rms_value, timestamp)

    def reset(self, sensor_name: Optional[str] = None):
        """
        重置指定传感器或所有传感器的状态
        """
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
    """单个传感器的状态管理与检测逻辑"""
    def __init__(self, sensor_name: str, config: Dict[str, Any]):
        self.sensor_name = sensor_name
        self.rms_threshold = config.get('rms_threshold', 0.3)
        self.consecutive_exceed_count = config.get('consecutive_exceed_count', 5)
        self.alarm_cooldown = config.get('alarm_cooldown', 30)
        self.window_size = config.get('window_size', 10)  # 保留，未使用

        # 运行时状态
        self.exceed_counter = 0
        self.alarm_triggered = False
        self.last_alarm_time = 0.0
        self.rms_history = deque(maxlen=100)  # 调试用

        logger.debug(f"[{sensor_name}] 初始化: 阈值={self.rms_threshold:.4f}g, "
                     f"连续次数={self.consecutive_exceed_count}, 冷却={self.alarm_cooldown}s")

    def update(self, rms_value: float, timestamp: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """更新状态并判断是否报警"""
        self.rms_history.append(rms_value)

        # 1. 超限判断与计数器更新
        if rms_value > self.rms_threshold:
            self.exceed_counter += 1
            logger.debug(f"[{self.sensor_name}] RMS超限: {rms_value:.4f} > {self.rms_threshold:.4f}, "
                         f"连续 {self.exceed_counter}/{self.consecutive_exceed_count}")
        else:
            if self.exceed_counter > 0:
                logger.debug(f"[{self.sensor_name}] RMS回落，计数器清零")
            self.exceed_counter = 0
            if self.alarm_triggered:
                self.alarm_triggered = False
                logger.info(f"[{self.sensor_name}] 螺栓松动报警解除")

        # 2. 报警触发条件
        alarm_info = None
        if (self.exceed_counter >= self.consecutive_exceed_count and 
            not self.alarm_triggered and 
            not self._in_cooldown(timestamp)):
            
            self.alarm_triggered = True
            self.last_alarm_time = timestamp
            alarm_info = {
                'fault_type': 'bolt_loosen',
                'sensor': self.sensor_name,
                'rms': rms_value,
                'threshold': self.rms_threshold,
                'exceed_count': self.exceed_counter,
                'message': (f"螺栓松动预警: {self.sensor_name} RMS连续{self.consecutive_exceed_count}次"
                            f"超过阈值{self.rms_threshold:.4f}g"),
                'timestamp': timestamp
            }
            logger.warning(alarm_info['message'])

        return self.alarm_triggered, alarm_info

    def _in_cooldown(self, current_time: float) -> bool:
        if self.alarm_cooldown <= 0:
            return False
        return (current_time - self.last_alarm_time) < self.alarm_cooldown

    def reset(self):
        self.exceed_counter = 0
        self.alarm_triggered = False
        self.last_alarm_time = 0.0
        self.rms_history.clear()