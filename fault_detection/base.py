from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
from loguru import logger as mylog

class BaseFaultDetector(ABC):
    """故障检测器抽象基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Args:
            name: 检测器实例名称（如 'bearing'）
            config: 该检测器的配置字典（包含 params 等）
        """
        self.name = name
        self.config = config
        self.params = config.get("params", {})
        self.mylog = mylog.bind(classname=self.__class__.__name__)
    
    @abstractmethod
    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        """
        处理传感器数据，判断是否发生故障
        
        Args:
            sensor_name: 传感器标识
            data_packet: 包含特征数据的字典，至少包含：
                - rms_value: float
                - fft_all: {"fft": array, "index": array} 或 None
                - timestamp: float
        
        Returns:
            (is_fault, extra_info)
            - is_fault: 是否触发报警
            - extra_info: 附加信息字典，若无则为 None
        """
        pass
    
    @abstractmethod
    def reset(self, sensor_name: Optional[str] = None):
        """重置检测器内部状态"""
        pass