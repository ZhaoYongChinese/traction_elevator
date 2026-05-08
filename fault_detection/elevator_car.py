from typing import Dict, Optional, Tuple, Any
import time
import numpy as np


class ElevatorCarFaultDetector:
    """轿厢故障分级检测器（最终版）
    1. 输入原始三轴振动信号
    2. 自动计算 RMS、peak_factor、impulse_factor、margin_factor
    3. 外部传入基准 rms_config，自动计算比值 rms_ratio
    4. 所有阈值 X/Y/Z 独立
    5. 连续 3 次异常 → 报警
    6. Z轴超标 = 轿架振动；X/Y轴超标 = 平稳度异常（无等级）
    """

    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        self.name = name
        self.default_params = config.get('params', {})
        self.sensors_config = config.get('parsed_sensors', [])

        # 三轴独立阈值
        self.default_pf_thresh_x = self.default_params.get('stability_pf_thresh_x', 5.0)
        self.default_pf_thresh_y = self.default_params.get('stability_pf_thresh_y', 5.0)
        self.default_pf_thresh_z = self.default_params.get('stability_pf_thresh_z', 5.0)

        self.default_if_thresh_x = self.default_params.get('stability_if_thresh_x', 6.0)
        self.default_if_thresh_y = self.default_params.get('stability_if_thresh_y', 6.0)
        self.default_if_thresh_z = self.default_params.get('stability_if_thresh_z', 6.0)

        self.default_mf_thresh_x = self.default_params.get('stability_mf_thresh_x', 7.0)
        self.default_mf_thresh_y = self.default_params.get('stability_mf_thresh_y', 7.0)
        self.default_mf_thresh_z = self.default_params.get('stability_mf_thresh_z', 7.0)

        # RMS 分级阈值（比值阈值）
        self.rms_level1_x = self.default_params.get('rms_level1_x', 4.0)
        self.rms_level1_y = self.default_params.get('rms_level1_y', 4.0)
        self.rms_level1_z = self.default_params.get('rms_level1_z', 4.0)

        self.trigger_times = 3  # 连续3次报警

        # 传感器配置
        self.sensor_configs = {}
        for sensor_cfg in self.sensors_config:
            sensor_name = sensor_cfg.get('name')
            if sensor_name:
                self.sensor_configs[sensor_name] = sensor_cfg

        # 状态计数器
        self.sensor_states: Dict[str, Dict] = {}
        for sensor_name in self.sensor_configs.keys():
            self.sensor_states[sensor_name] = {
                'current_level': 0,
                'continuous_count': 0,
                'alarm_triggered': False,
                'alarm_type': ''  # 存报警类型：平稳度异常 / 轿架振动
            }

    # 指标计算（含 RMS）
    def _calculate_single_axis_features(self, signal):
        sig = np.asarray(signal, dtype=np.float32).ravel()
        if len(sig) == 0:
            return 0.0, 0.0, 0.0, 0.0

        mean_abs = np.mean(np.abs(sig))
        rms = np.sqrt(np.mean(sig ** 2))
        peak = np.max(np.abs(sig))
        root_abs_mean = (np.mean(np.sqrt(np.abs(sig)))) ** 2

        pf = peak / rms if rms != 0 else 0.0
        imp = peak / mean_abs if mean_abs != 0 else 0.0
        mar = peak / root_abs_mean if root_abs_mean != 0 else 0.0
        return pf, imp, mar, rms

    def calculate_3axis_features(self, x_sig, y_sig, z_sig, rms_config):
        # 计算三轴所有指标
        pf_x, imp_x, mar_x, rms_x = self._calculate_single_axis_features(x_sig)
        pf_y, imp_y, mar_y, rms_y = self._calculate_single_axis_features(y_sig)
        pf_z, imp_z, mar_z, rms_z = self._calculate_single_axis_features(z_sig)

        # 自动计算 RMS 比值
        rms_ratio_x = rms_x / rms_config['X'] if rms_config['X'] != 0 else 0
        rms_ratio_y = rms_y / rms_config['Y'] if rms_config['Y'] != 0 else 0
        rms_ratio_z = rms_z / rms_config['Z'] if rms_config['Z'] != 0 else 0

        return {
            "peak_factor": {"X": pf_x, "Y": pf_y, "Z": pf_z},
            "impulse_factor": {"X": imp_x, "Y": imp_y, "Z": imp_z},
            "margin_factor": {"X": mar_x, "Y": mar_y, "Z": mar_z},
            "rms_ratio": {"X": rms_ratio_x, "Y": rms_ratio_y, "Z": rms_ratio_z},
        }

    # 单轴异常判断：3个指标中≥2个超过【基础阈值】即可
    def _check_axis_abnormal(self, axis: str, pf: Dict, imp: Dict, mar: Dict) -> bool:
        pf_val = pf[axis]
        imp_val = imp[axis]
        mar_val = mar[axis]

        if axis == "X":
            pf_t = self.default_pf_thresh_x
            if_t = self.default_if_thresh_x
            mf_t = self.default_mf_thresh_x
        elif axis == "Y":
            pf_t = self.default_pf_thresh_y
            if_t = self.default_if_thresh_y
            mf_t = self.default_mf_thresh_y
        else:
            pf_t = self.default_pf_thresh_z
            if_t = self.default_if_thresh_z
            mf_t = self.default_mf_thresh_z

        exceed = int(pf_val > pf_t) + int(imp_val > if_t) + int(mar_val > mf_t)
        return exceed >= 2

    # ===================== ✅ 这里只改了判断逻辑 =====================
    def _get_alarm_type(self, pf, imp, mar, rms_ratio):
        """
        新逻辑：
        1. Z轴 RMS超标 + Z轴指标≥2个超标 → 轿架振动
        2. X/Y 任意RMS超标 + X/Y 任意指标≥2个超标 → 平稳度异常
        3. 都不满足 → 正常
        """
        # 取RMS比值
        rx = rms_ratio["X"]
        ry = rms_ratio["Y"]
        rz = rms_ratio["Z"]

        # Z轴报警（优先级最高）
        z_rms_ok = rz >= self.rms_level1_z
        z_feat_ok = self._check_axis_abnormal("Z", pf, imp, mar)
        if z_rms_ok and z_feat_ok:
            return "轿架振动"

        # X/Y 报警
        xy_rms_ok = (rx >= self.rms_level1_x) or (ry >= self.rms_level1_y)
        xy_feat_ok = self._check_axis_abnormal("X", pf, imp, mar) or self._check_axis_abnormal("Y", pf, imp, mar)
        if xy_rms_ok and xy_feat_ok:
            return "平稳度异常"

        # 正常
        return ""

    # =================================================================

    # 主更新接口
    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        sensor_cfg = self.sensor_configs.get(sensor_name)
        if not sensor_cfg:
            return False, None

        state = self.sensor_states[sensor_name]

        x = data_packet.get("x", [])
        y = data_packet.get("y", [])
        z = data_packet.get("z", [])
        rms_config = data_packet.get("rms_config", {"X": 1, "Y": 1, "Z": 1})

        # 自动计算所有指标 + RMS 比值
        feats = self.calculate_3axis_features(x, y, z, rms_config)
        pf = feats["peak_factor"]
        imp = feats["impulse_factor"]
        mar = feats["margin_factor"]
        rms_ratio = feats["rms_ratio"]

        # ===================== 新逻辑调用 =====================
        alarm_type = self._get_alarm_type(pf, imp, mar, rms_ratio)
        is_abnormal = len(alarm_type) > 0
        # ======================================================

        # 连续计数逻辑（完全不变）
        if is_abnormal:
            if alarm_type == state["alarm_type"]:
                state["continuous_count"] += 1
            else:
                state["alarm_type"] = alarm_type
                state["continuous_count"] = 1
        else:
            state["continuous_count"] = 0
            state["alarm_type"] = ""
            state["alarm_triggered"] = False

        alarm = None
        if state["continuous_count"] >= self.trigger_times and not state["alarm_triggered"]:
            state["alarm_triggered"] = True
            alarm = {
                "alarm_type": state["alarm_type"],
                "sensor": sensor_name,
                "rms_ratio_X": round(rms_ratio["X"], 2),
                "rms_ratio_Y": round(rms_ratio["Y"], 2),
                "rms_ratio_Z": round(rms_ratio["Z"], 2),
                "msg": f"连续{self.trigger_times}次异常 → {state['alarm_type']}"
            }

        return state["alarm_triggered"], alarm


# ------------------- 测试：100% 报警 -------------------
if __name__ == '__main__':
    config = {
        "params": {
            "stability_pf_thresh_x": 2, "stability_pf_thresh_y": 2, "stability_pf_thresh_z": 8,
            "stability_if_thresh_x": 2, "stability_if_thresh_y": 2, "stability_if_thresh_z": 8,
            "stability_mf_thresh_x": 2, "stability_mf_thresh_y": 2, "stability_mf_thresh_z": 8,
            "rms_level1_x": 4, "rms_level1_y": 4, "rms_level1_z": 20,
        },
        "parsed_sensors": [{"name": "car_sensor"}]
    }

    detector = ElevatorCarFaultDetector("轿厢检测", config)

    test_data = {
        "x": [0]*200 + [10000],
        "y": [0]*200 + [12000],
        "z": [0]*200 + [15000],
        "rms_config": {"X": 10, "Y": 10, "Z": 100}
    }

    print("1:", detector.update("car_sensor", test_data))
    print("2:", detector.update("car_sensor", test_data))
    print("3:", detector.update("car_sensor", test_data))