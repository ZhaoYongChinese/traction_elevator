from typing import Dict, List, Tuple, Any, Optional
from loguru import logger as mylog
from .base import BaseFaultDetector
from .bearing_fault import BearingFaultDetector
from .bolt_loosen import BoltLoosenDetector
from .rope_fault import RopeElongationDetector, RopeFaultDetector
from .motor_fault import MotorFaultDetector
from .elevator_car import ElevatorCarFaultDetector


class FaultManager:
    def __init__(self, global_config: Dict):
        self.global_config = global_config
        self.detectors: Dict[str, BaseFaultDetector] = {}      # name -> detector instance
        self.sensor_map: Dict[str, List[str]] = {}            # sensor_name -> list of detector names
        self._init_detectors()

    def _init_detectors(self):
        det_cfg = self.global_config.get("fault_detectors", {})
        for det_name, det_conf in det_cfg.items():
            det_type = det_conf.get("type")
            sensors_raw = det_conf.get("sensors", [])

            # 解析传感器配置，统一为字典列表（每个字典至少包含 'name'）
            parsed_sensors = []
            for item in sensors_raw:
                if isinstance(item, str):
                    parsed_sensors.append({"name": item})
                elif isinstance(item, dict):
                    # 确保至少有 name 字段
                    if "name" in item:
                        parsed_sensors.append(item)
                    else:
                        mylog.error(f"检测器 {det_name} 的传感器配置缺少 'name' 字段: {item}")
                else:
                    mylog.error(f"检测器 {det_name} 的传感器配置格式错误: {item}")

            # 将解析后的传感器列表存入 det_conf，供检测器使用
            det_conf["parsed_sensors"] = parsed_sensors

            # 根据类型实例化检测器
            if det_type == "bearing":
                detector = BearingFaultDetector(det_name, det_conf, self.global_config)
            elif det_type == "bolt_loosen":
                detector = BoltLoosenDetector(det_name, det_conf)
            elif det_type == "rope":
                detector = RopeFaultDetector(det_name, det_conf, self.global_config)
            elif det_type == "motor":
                detector = MotorFaultDetector(det_name, det_conf, self.global_config)
            elif det_type == "elevator_car":
                detector = ElevatorCarFaultDetector(det_name, det_conf, self.global_config)

            else:
                mylog.error(f"Unknown detector type: {det_type}")
                continue

            self.detectors[det_name] = detector

            # 建立传感器名称到检测器的映射
            for sensor_info in parsed_sensors:
                sensor_name = sensor_info["name"]
                if sensor_name not in self.sensor_map:
                    self.sensor_map[sensor_name] = []
                self.sensor_map[sensor_name].append(det_name)

            mylog.info(f"Initialized detector '{det_name}' (type: {det_type}) for sensors: "
                       f"{[s['name'] for s in parsed_sensors]}")

    def process(self, sensor_name: str, data_packet: Dict) -> List[Tuple[str, bool, Dict]]:
        """
        处理传感器数据，返回所有触发的故障
        返回格式: [(detector_name, is_fault, extra_info), ...]
        """
        results = []
        detector_names = self.sensor_map.get(sensor_name, [])
        for det_name in detector_names:
            detector = self.detectors.get(det_name)
            if detector is None:
                continue
            try:
                is_fault, extra = detector.update(sensor_name, data_packet)
                if is_fault:
                    results.append((det_name, is_fault, extra))
            except Exception as e:
                mylog.error(f"Error in detector '{det_name}' for sensor '{sensor_name}': {e}")
        return results

    def reset_detector(self, detector_name: str, sensor_name: Optional[str] = None):
        if detector_name in self.detectors:
            self.detectors[detector_name].reset(sensor_name)

    def reset_all(self):
        for detector in self.detectors.values():
            detector.reset()