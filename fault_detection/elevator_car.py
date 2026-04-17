# elevator_car.py
import time
from typing import Dict, Optional, Tuple, Any
from loguru import logger


class ElevatorCarFaultDetector:
    """
    轿厢平稳度异常检测器（边缘端简化版）
    仅诊断：运行晃动 —— 基于 X/Y 轴时域指标（峰值因子、脉冲因子、裕度因子）连续超阈值
    """

    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        self.name = name
        self.default_params = config.get('params', {})
        self.sensors_config = config.get('parsed_sensors', [])

        # 默认阈值参数
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

        logger.info(f"[{self.name}] 轿厢平稳度检测器初始化完成，管理传感器: {list(self._states.keys())}")

    def _init_states(self):
        for sensor_name, sensor_cfg in self.sensor_configs.items():
            # 合并参数：传感器独立配置优先
            pf_thresh = sensor_cfg.get('stability_pf_thresh', self.default_pf_thresh)
            if_thresh = sensor_cfg.get('stability_if_thresh', self.default_if_thresh)
            mf_thresh = sensor_cfg.get('stability_mf_thresh', self.default_mf_thresh)
            trigger_count = sensor_cfg.get('trigger_count', self.default_trigger_count)
            cooldown = sensor_cfg.get('alarm_cooldown', self.default_alarm_cooldown)

            self._states[sensor_name] = _SensorState(
                sensor_name, pf_thresh, if_thresh, mf_thresh, trigger_count, cooldown
            )

    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        """
        处理传感器数据包，要求包含以下字段：
        - peak_factor: dict, 至少含 'X', 'Y'
        - impulse_factor: dict, 至少含 'X', 'Y'
        - margin_factor: dict, 至少含 'X', 'Y'
        - timestamp (可选)
        """
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
            logger.error(f"[{self.name}] 数据包缺少必要的时域指标字段 (peak_factor/impulse_factor/margin_factor)")
            return False, None

        timestamp = data_packet.get('timestamp', time.time())
        return state.update(pf, imp, mar, timestamp)

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
    """单个传感器的平稳度检测状态机"""
    def __init__(self, sensor_name: str, pf_thresh: float, if_thresh: float, mf_thresh: float,
                 trigger_count: int, cooldown: int):
        self.sensor_name = sensor_name
        self.pf_thresh = pf_thresh
        self.if_thresh = if_thresh
        self.mf_thresh = mf_thresh
        self.trigger_count = trigger_count
        self.cooldown = cooldown

        self.exceed_counter = 0
        self.alarm_triggered = False
        self.last_alarm_time = 0.0

        logger.debug(f"[{sensor_name}] 平稳度检测参数: PF>{pf_thresh}, IF>{if_thresh}, MF>{mf_thresh}, "
                     f"连续次数={trigger_count}, 冷却={cooldown}s")

    def _is_abnormal(self, pf: Dict, imp: Dict, mar: Dict) -> bool:
        """
        判断当前帧是否异常：
        X 轴或 Y 轴的三个指标中，至少两个超阈值即认为异常
        """
        x_pf = pf.get('X', 0)
        y_pf = pf.get('Y', 0)
        x_imp = imp.get('X', 0)
        y_imp = imp.get('Y', 0)
        x_mar = mar.get('X', 0)
        y_mar = mar.get('Y', 0)

        score_x = (x_pf > self.pf_thresh) + (x_imp > self.if_thresh) + (x_mar > self.mf_thresh)
        score_y = (y_pf > self.pf_thresh) + (y_imp > self.if_thresh) + (y_mar > self.mf_thresh)
        return (score_x >= 2) or (score_y >= 2)

    def update(self, pf: Dict, imp: Dict, mar: Dict, timestamp: float) -> Tuple[bool, Optional[Dict]]:
        abnormal = self._is_abnormal(pf, imp, mar)

        if abnormal:
            self.exceed_counter += 1
            logger.debug(f"[{self.sensor_name}] 时域指标异常，连续 {self.exceed_counter}/{self.trigger_count}")
        else:
            if self.exceed_counter > 0:
                logger.debug(f"[{self.sensor_name}] 指标恢复正常，计数器清零")
            self.exceed_counter = 0
            if self.alarm_triggered:
                self.alarm_triggered = False
                logger.info(f"[{self.sensor_name}] 平稳度异常报警解除")

        alarm_info = None
        if (self.exceed_counter >= self.trigger_count and
            not self.alarm_triggered and
            not self._in_cooldown(timestamp)):

            self.alarm_triggered = True
            self.last_alarm_time = timestamp
            alarm_info = {
                'fault_type': 'elevator_car_stability',
                'sensor': self.sensor_name,
                'message': f"轿厢平稳度异常预警: {self.sensor_name} 时域指标连续{self.trigger_count}次超阈值",
                'pf_thresh': self.pf_thresh,
                'if_thresh': self.if_thresh,
                'mf_thresh': self.mf_thresh,
                'exceed_count': self.exceed_counter,
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