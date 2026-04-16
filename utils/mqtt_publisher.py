import json
import os
import time
import paho.mqtt.client as mqtt
from loguru import logger as mylog


class MQTTPublisher:
    def __init__(self):
        self.mylog = mylog.bind(classname=self.__class__.__name__)
        self.mq_p_init = False
        self.server_ip = None
        self.sn_value = None
        config_path = '/userdata/config.json'
        sn_path = '/deviceSN'

        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"文件不存在: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.server_ip = data["server"]

            if not os.path.exists(sn_path):
                raise FileNotFoundError(f"文件不存在: {sn_path}")
            with open(sn_path, 'r', encoding='utf-8') as file:
                first_line = file.readline().strip()
                self.sn_value = first_line

        except FileNotFoundError as e:
            mylog.error(f"❌ 文件错误: {e}")
        except json.JSONDecodeError as e:
            mylog.error(f"❌ JSON格式错误: {e}")
        except Exception as e:
            mylog.error(f"❌ 读取文件时发生错误: {e}")

        if self.server_ip and self.sn_value:
            self.mq_p_init = True
            self.broker = self.server_ip
            self.port = 1883
            self.client_id = "WG3588_pd_" + self.sn_value
            self.client = mqtt.Client(client_id=self.client_id)
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            mylog.info(f"服务器主机IP为：{self.broker}, 当前客户端ID为：{self.client_id}")
        else:
            mylog.error("从板端获取服务器IP或SN值失败，无法发送消息。")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.mylog.info(f"成功连接到 MQTT 代理: {self.broker}:{self.port}")
        else:
            self.mylog.error(f"连接失败，错误码: {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.mylog.error(f"MQTT 连接断开，错误码: {rc}")
        if rc != 0:
            self.mylog.warning("尝试重新连接...")
            self.connect()

    def connect(self):
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            self.mylog.info("MQTT 客户端正在链接...")
        except Exception as e:
            self.mylog.error(f"连接 MQTT 代理失败: {e}")
            time.sleep(5)
            self.connect()

    def publish(self, topic, payload, qos=1, retain=False):
        try:
            payload_json = json.dumps(payload)
            self.client.publish(topic, payload=payload_json, qos=qos, retain=retain)
            self.mylog.debug(f"MQTT 发布消息到 {topic}: {payload}")
        except Exception as e:
            self.mylog.error(f"MQTT 发布失败: {e}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        self.mylog.info("MQTT 客户端已断开连接")