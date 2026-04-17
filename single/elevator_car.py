# elevator_car.py
import time
from typing import Dict, Optional, Tuple, Any
from loguru import logger


class ElevatorCarFaultDetector:
    """
    轿厢故障分级检测器（基于三轴时域指标倍数）
    故障分级：
    - 平稳度异常：至少2个方向，在3个时域指标中至少2个超过对应阈值的 2 倍
    - 轿架振动：至少2个方向，在3个时域指标中至少2个超过对应阈值的 4 倍
    倍数分级：4倍以上同时满足轿架振动条件，优先报高级别故障。
    """

    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        self.name = name
        self.default_params = config.get('params', {})
        self.sensors_config = config.get('parsed_sensors', [])

        # 默认阈值参数（各轴独立）
        self.default_pf_thresh = self.default_params.get('stability_pf_thresh', 5.0)
        self.default_if_thresh = self.default_params.get('stability_if_thresh', 6.0)
        self.default_mf_thresh = self.default_params.get('stability_mf_thresh', 7.0)

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

        logger.info(f"[{self.name}] 轿厢分级检测器初始化完成，管理传感器: {list(self._states.keys())}")

    def _init_states(self):
        for sensor_name, sensor_cfg in self.sensor_configs.items():
            # 各轴阈值，可从传感器独立配置覆盖，否则使用默认值
            pf_thresh = {
                'X': sensor_cfg.get('pf_thresh_X', self.default_pf_thresh),
                'Y': sensor_cfg.get('pf_thresh_Y', self.default_pf_thresh),
                'Z': sensor_cfg.get('pf_thresh_Z', self.default_pf_thresh)
            }
            if_thresh = {
                'X': sensor_cfg.get('if_thresh_X', self.default_if_thresh),
                'Y': sensor_cfg.get('if_thresh_Y', self.default_if_thresh),
                'Z': sensor_cfg.get('if_thresh_Z', self.default_if_thresh)
            }
            mf_thresh = {
                'X': sensor_cfg.get('mf_thresh_X', self.default_mf_thresh),
                'Y': sensor_cfg.get('mf_thresh_Y', self.default_mf_thresh),
                'Z': sensor_cfg.get('mf_thresh_Z', self.default_mf_thresh)
            }

            trigger_count = sensor_cfg.get('trigger_count', self.default_trigger_count)
            cooldown = sensor_cfg.get('alarm_cooldown', self.default_alarm_cooldown)

            self._states[sensor_name] = _SensorState(
                sensor_name, pf_thresh, if_thresh, mf_thresh, trigger_count, cooldown
            )

    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        state = self._states.get(sensor_name)
        if state is None:
            logger.warning(f"[{self.name}] 收到未注册传感器 '{sensor_name}' 的数据，忽略")
            return False, None

        # 仅处理稳态数据（可选）
        if data_packet.get('running_state') not in ('steady', None):
            return False, None

        pf = data_packet.get('peak_factor', {})
        imp = data_packet.get('impulse_factor', {})
        mar = data_packet.get('margin_factor', {})

        if not all([pf, imp, mar]):
            logger.error(f"[{self.name}] 数据包缺少必要的时域指标字段")
            return False, None

        timestamp = data_packet.get('timestamp', time.time())
        return state.update(pf, imp, mar, timestamp)

    def reset(self, sensor_name: Optional[str] = None):
        if sensor_name:
            state = self._states.get(sensor_name)
            if state:
                state.reset()
        else:
            for state in self._states.values():
                state.reset()


class _SensorState:
    """单个传感器的分级状态机"""
    def __init__(self, sensor_name: str,
                 pf_thresh: Dict[str, float],
                 if_thresh: Dict[str, float],
                 mf_thresh: Dict[str, float],
                 trigger_count: int, cooldown: int):
        self.sensor_name = sensor_name
        self.pf_thresh = pf_thresh
        self.if_thresh = if_thresh
        self.mf_thresh = mf_thresh
        self.trigger_count = trigger_count
        self.cooldown = cooldown

        self.exceed_counter = 0
        self.current_fault_level = 0          # 0:正常, 1:平稳度异常(2倍), 2:轿架振动(4倍)
        self.alarm_triggered = False
        self.last_alarm_time = 0.0

        logger.debug(f"[{sensor_name}] 分级检测参数: PF阈值={pf_thresh}, IF阈值={if_thresh}, MF阈值={mf_thresh}, "
                     f"连续次数={trigger_count}, 冷却={cooldown}s")

    def _check_direction_anomaly(self, axis: str, pf: Dict, imp: Dict, mar: Dict, multiplier: float) -> bool:
        """检查某个方向在指定倍数下是否异常（至少两个指标超过阈值*倍数）"""
        pf_val = pf.get(axis, 0)
        imp_val = imp.get(axis, 0)
        mar_val = mar.get(axis, 0)

        thresh_pf = self.pf_thresh[axis] * multiplier
        thresh_if = self.if_thresh[axis] * multiplier
        thresh_mf = self.mf_thresh[axis] * multiplier

        score = (pf_val > thresh_pf) + (imp_val > thresh_if) + (mar_val > thresh_mf)
        return score >= 2

    def _count_abnormal_directions(self, pf: Dict, imp: Dict, mar: Dict, multiplier: float) -> int:
        """统计在指定倍数下有多少个方向异常"""
        count = 0
        for axis in ['X', 'Y', 'Z']:
            if self._check_direction_anomaly(axis, pf, imp, mar, multiplier):
                count += 1
        return count

    def _get_fault_level(self, pf: Dict, imp: Dict, mar: Dict) -> int:
        """
        根据数据判断故障级别：
        1级（平稳度异常）：至少2个方向满足2倍条件
        2级（轿架振动）：至少2个方向满足4倍条件
        """
        if self._count_abnormal_directions(pf, imp, mar, 4.0) >= 2:
            return 2
        elif self._count_abnormal_directions(pf, imp, mar, 2.0) >= 2:
            return 1
        else:
            return 0

    def update(self, pf: Dict, imp: Dict, mar: Dict, timestamp: float) -> Tuple[bool, Optional[Dict]]:
        current_level = self._get_fault_level(pf, imp, mar)

        if current_level > 0 and current_level == self.current_fault_level:
            self.exceed_counter += 1
            logger.debug(f"[{self.sensor_name}] 故障级别 {current_level} 持续，"
                         f"连续 {self.exceed_counter}/{self.trigger_count}")
        else:
            if self.exceed_counter > 0:
                logger.debug(f"[{self.sensor_name}] 故障级别变化或恢复，计数器清零")
            self.exceed_counter = 0
            self.current_fault_level = current_level

            if self.alarm_triggered and current_level == 0:
                self.alarm_triggered = False
                logger.info(f"[{self.sensor_name}] 轿厢故障报警解除")

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
                'level': self.current_fault_level,
                'message': (f"轿厢{self._level_to_chinese(fault_type)}预警: {self.sensor_name} "
                            f"连续{self.trigger_count}次触发"),
                'exceed_count': self.exceed_counter,
                'timestamp': timestamp
            }
            logger.warning(alarm_info['message'])

        return self.alarm_triggered, alarm_info

    def _level_to_fault_type(self, level: int) -> str:
        return {
            1: 'elevator_car_stability',
            2: 'elevator_car_frame_vibration'
        }.get(level, 'unknown')

    def _level_to_chinese(self, fault_type: str) -> str:
        return {
            'elevator_car_stability': '平稳度异常',
            'elevator_car_frame_vibration': '轿架振动'
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