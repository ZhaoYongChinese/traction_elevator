# rope_fault.py
import time
from typing import Dict, Optional, Tuple, Any
from loguru import logger


class RopeFaultDetector:
    """
    钢丝绳故障分级检测器（基于 RMS 倍数）
    故障分级：
    - 张力不均：RMS 达到基线的 4 倍以上
    - 打滑：RMS 达到基线的 6 倍以上
    - 磨损/断裂：RMS 达到基线的 8 倍以上
    注意：倍数区间为左闭右开，高倍数自动覆盖低级别故障（取最高级）。
    """

    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        self.name = name
        self.default_params = config.get('params', {})
        self.sensors_config = config.get('parsed_sensors', [])

        # 默认倍数阈值
        self.default_ratio_tension = self.default_params.get('ratio_tension', 4.0)
        self.default_ratio_slip = self.default_params.get('ratio_slip', 6.0)
        self.default_ratio_wear = self.default_params.get('ratio_wear', 8.0)
        self.default_trigger_count = self.default_params.get('trigger_count', 3)
        self.default_alarm_cooldown = self.default_params.get('alarm_cooldown', 30)

        # 传感器独立配置
        self.sensor_configs = {}
        for sensor_cfg in self.sensors_config:
            sensor_name = sensor_cfg.get('name')
            if sensor_name:
                self.sensor_configs[sensor_name] = sensor_cfg

        # 运行时状态
        self._states: Dict[str, _SensorState] = {}
        self._init_states()

        logger.info(f"[{self.name}] 钢丝绳 RMS 分级检测器初始化完成，管理传感器: {list(self._states.keys())}")

    def _init_states(self):
        for sensor_name, sensor_cfg in self.sensor_configs.items():
            baseline_rms = sensor_cfg.get('baseline_rms')
            if baseline_rms is None:
                logger.error(f"[{self.name}] 传感器 '{sensor_name}' 缺少 'baseline_rms' 配置，将无法工作")
                continue

            ratio_tension = sensor_cfg.get('ratio_tension', self.default_ratio_tension)
            ratio_slip = sensor_cfg.get('ratio_slip', self.default_ratio_slip)
            ratio_wear = sensor_cfg.get('ratio_wear', self.default_ratio_wear)
            trigger_count = sensor_cfg.get('trigger_count', self.default_trigger_count)
            cooldown = sensor_cfg.get('alarm_cooldown', self.default_alarm_cooldown)

            self._states[sensor_name] = _SensorState(
                sensor_name, baseline_rms,
                ratio_tension, ratio_slip, ratio_wear,
                trigger_count, cooldown
            )

    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        """
        处理传感器数据包，要求包含以下字段：
        - rms 或 rms_g：float，当前 RMS 值（单位：g）
        - timestamp (可选)
        """
        state = self._states.get(sensor_name)
        if state is None:
            logger.warning(f"[{self.name}] 收到未注册传感器 '{sensor_name}' 的数据，忽略")
            return False, None

        # 仅处理稳态数据（可选）
        if data_packet.get('running_state') not in ('steady', None):
            return False, None

        rms_value = data_packet.get('rms') or data_packet.get('rms_g')
        if rms_value is None:
            logger.error(f"[{self.name}] 数据包缺少 RMS 字段 ('rms' 或 'rms_g')")
            return False, None

        timestamp = data_packet.get('timestamp', time.time())
        return state.update(rms_value, timestamp)

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
    """单个传感器的 RMS 分级状态机"""
    def __init__(self, sensor_name: str, baseline_rms: float,
                 ratio_tension: float, ratio_slip: float, ratio_wear: float,
                 trigger_count: int, cooldown: int):
        self.sensor_name = sensor_name
        self.baseline_rms = baseline_rms
        self.ratio_tension = ratio_tension
        self.ratio_slip = ratio_slip
        self.ratio_wear = ratio_wear
        self.trigger_count = trigger_count
        self.cooldown = cooldown

        # 确保倍数递增
        assert ratio_tension <= ratio_slip <= ratio_wear, "倍数阈值必须递增"

        self.exceed_counter = 0
        self.current_fault_level = 0          # 0:正常, 1:张力不均, 2:打滑, 3:磨损/断裂
        self.alarm_triggered = False
        self.last_alarm_time = 0.0

        logger.debug(f"[{sensor_name}] RMS分级检测: 基线={baseline_rms:.4f}g, "
                     f"倍数阈值: 张力={ratio_tension}x, 打滑={ratio_slip}x, 磨损={ratio_wear}x, "
                     f"连续次数={trigger_count}, 冷却={cooldown}s")

    def _get_fault_level(self, ratio: float) -> int:
        if ratio >= self.ratio_wear:
            return 3
        elif ratio >= self.ratio_slip:
            return 2
        elif ratio >= self.ratio_tension:
            return 1
        else:
            return 0

    def _level_to_fault_type(self, level: int) -> str:
        return {
            1: 'tension_imbalance',
            2: 'slippage',
            3: 'wear_broken'
        }.get(level, 'unknown')

    def update(self, rms_value: float, timestamp: float) -> Tuple[bool, Optional[Dict]]:
        ratio = rms_value / (self.baseline_rms + 1e-10)
        current_level = self._get_fault_level(ratio)

        if current_level > 0 and current_level == self.current_fault_level:
            self.exceed_counter += 1
            logger.debug(f"[{self.sensor_name}] RMS倍数 {ratio:.2f} 持续触发级别 {current_level}，"
                         f"连续 {self.exceed_counter}/{self.trigger_count}")
        else:
            if self.exceed_counter > 0:
                logger.debug(f"[{self.sensor_name}] 故障级别变化或恢复，计数器清零")
            self.exceed_counter = 0
            self.current_fault_level = current_level

            if self.alarm_triggered and current_level == 0:
                self.alarm_triggered = False
                logger.info(f"[{self.sensor_name}] 钢丝绳故障报警解除")

        alarm_info = None
        if (self.exceed_counter >= self.trigger_count and
            not self.alarm_triggered and
            not self._in_cooldown(timestamp)):

            self.alarm_triggered = True
            self.last_alarm_time = timestamp
            fault_type = self._level_to_fault_type(self.current_fault_level)
            alarm_info = {
                'fault_type': fault_type,
                'sensor': self.sensor_name,
                'rms': rms_value,
                'baseline_rms': self.baseline_rms,
                'ratio': ratio,
                'threshold_ratio': self._get_threshold_for_level(self.current_fault_level),
                'exceed_count': self.exceed_counter,
                'message': (f"钢丝绳{self._level_to_chinese(fault_type)}预警: {self.sensor_name} "
                            f"RMS为基线的{ratio:.2f}倍，连续{self.trigger_count}次超阈值"),
                'timestamp': timestamp
            }
            logger.warning(alarm_info['message'])

        return self.alarm_triggered, alarm_info

    def _get_threshold_for_level(self, level: int) -> float:
        if level == 1:
            return self.ratio_tension
        elif level == 2:
            return self.ratio_slip
        elif level == 3:
            return self.ratio_wear
        return 0.0

    def _level_to_chinese(self, fault_type: str) -> str:
        return {
            'tension_imbalance': '张力不均',
            'slippage': '打滑',
            'wear_broken': '磨损/断裂'
        }.get(fault_type, fault_type)

    def _in_cooldown(self, current_time: float) -> bool:
        if self.cooldown <= 0:
            return False
        return (current_time - self.last_alarm_time) < self.cooldown

    def reset(self):
        self.exceed_counter = 0
        self.current_fault_level = 0
        self.alarm_triggered = False
        self.last_alarm_time = 0.0