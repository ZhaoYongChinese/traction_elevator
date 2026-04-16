import datetime
import json
import os
import sys
import time
import zmq
from loguru import logger as mylog
from ruamel.yaml import YAML
import requests
import paho.mqtt.client as mqtt
from threading import Lock, Timer
import multiprocessing

# 导入故障检测模块
from fault_detection.manager import FaultManager
# 趋势预测类（保持不变，可放在单独模块中）
from utils.trend_predict import TrendPredict
# MQTT 和报警节流类（保持不变）
from utils.mqtt_publisher import MQTTPublisher
from utils.sensor_alarm_throttler import SensorAlarmThrottler

# 辅助函数（与原代码相同）
def real_path(*path):
    file_path = sys.path[0]
    target_path = os.path.join(file_path, *path)
    return target_path

def make_logpath(logdir_name):
    log_dir_path = real_path(logdir_name)
    log_name = 'RMS_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + '.log'
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    log_path = real_path(logdir_name, log_name)
    return log_path

def SetUp_logger(log_path, level="INFO"):
    mylog.remove()
    mylog.add(log_path, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {process} | {module} | {function}:{line} - {message}", level=level, rotation="10 MB", enqueue=True, mode="a")
    mylog.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>{process}</cyan> | <cyan>{module}:{function}:{line}</cyan> - <level>{message}</level>", level="DEBUG")
    mylog.info(f"日志初始化成功，日志级别为 {level}。")

if __name__ == '__main__':
    VERSION = "0.4.0-20260101"
    log_path = make_logpath("RMS_LOG")
    SetUp_logger(log_path)
    mylog.info(f"版本号：{VERSION}")

    # 加载配置
    yaml = YAML(typ='safe', pure=True)
    with open(real_path("middleware_config.yml"), 'r') as file:
        cfg = yaml.load(file)
    if cfg["log_level"] != "INFO":
        SetUp_logger(log_path, cfg["log_level"])

    # ZMQ 接收
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://localhost:33333")
    mylog.info("ZMQ Socket 已绑定，等待数据...")

    # MQTT 发布器
    publisher = MQTTPublisher()
    if publisher.mq_p_init:
        publisher.connect()
        mylog.info("MQTT 初始化成功。")
    else:
        mylog.error("MQTT 初始化失败，将无法发送趋势预测。")

    # 报警节流器
    if publisher.mq_p_init:
        url = "http://" + publisher.server_ip + "/index.php?m=Api&c=alert&a=postAlert"
        fd_alarm = SensorAlarmThrottler(alarm_url=url, config=cfg, sever_ip=publisher.server_ip, mid_value=publisher.sn_value)
    else:
        fd_alarm = None

    # 趋势预测工具
    tp_tool = TrendPredict(window_size=cfg["PREDICT_WINDOW"])
    mylog.info("趋势预测初始化完成。")

    # 故障管理器
    fault_manager = FaultManager(cfg)
    mylog.info("故障管理器初始化完成。")

    while True:
        data = socket.recv_pyobj()
        sensor_name = data["sensor_name"]
        rms_value = data["rms_value"]
        fft_all = data.get("fft_all")
        timestamp = data["timestamp"]

        # 趋势预测并发送
        pre_value = tp_tool.update(sensor_name, rms_value)
        if publisher.mq_p_init:
            publisher.publish("pre_" + sensor_name, {"ori_val": rms_value, "pre_val": pre_value})

        # 故障诊断
        fault_results = fault_manager.process(sensor_name, data)
        for det_name, is_fault, extra in fault_results:
            # 根据检测器名称决定报警码（简单映射，可在配置中更精细化）
            # 这里假设检测器名称与报警码前缀有关联，或从 extra 中获取 fault_type
            fault_type = extra.get("fault_type", det_name)
            # 构建用于报警的传感器标识（可附加故障类型后缀）
            alarm_sensor_id = f"{sensor_name}_{fault_type}"
            
            mylog.info(f"故障触发: sensor={sensor_name}, detector={det_name}, extra={extra}")
            
            if fd_alarm:
                # 报警节流处理（注意原 process_sensor_data 参数可能需要调整）
                fd_alarm.process_sensor_data(alarm_sensor_id, timestamp)