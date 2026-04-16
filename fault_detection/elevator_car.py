import numpy as np
import json
import os
from typing import Dict, Optional, Tuple, Any, List
from collections import deque
from scipy.signal import find_peaks, coherence
from .base import BaseFaultDetector


class ElevatorCarFaultDetector(BaseFaultDetector):
    """
    轿厢大类故障检测器，包含：
    - 轿架振动异常（固有频率下降）
    - 轿厢平稳度异常（运行晃动）
    - 摩擦噪声异常（麦克风特征）
    - 水平度偏差（倾角仪长期监测）
    """

    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        super().__init__(name, config)

        # 通用参数
        self.window_size = self.params.get("window_size", 10)
        self.trigger_count = self.params.get("trigger_count", 3)

        # 轿架振动异常阈值
        self.freq_shift_ratio = self.params.get("freq_shift_ratio", 0.05)

        # 平稳度异常阈值
        self.stab_pf_thresh = self.params.get("stability_pf_thresh", 5.0)
        self.stab_if_thresh = self.params.get("stability_if_thresh", 6.0)
        self.stab_mf_thresh = self.params.get("stability_mf_thresh", 7.0)
        self.stab_low_energy_ratio = self.params.get("stability_low_energy_ratio", 3.0)

        # 摩擦噪声阈值
        self.noise_band_ratio = self.params.get("noise_band_energy_ratio", 3.0)
        self.noise_coherence_thresh = self.params.get("noise_coherence_thresh", 0.6)

        # 水平度偏差阈值
        self.level_angle_thresh = self.params.get("level_angle_thresh", 0.5)
        self.level_duration_thresh = self.params.get("level_duration_thresh", 30)

        # 传感器配置解析
        self.sensor_configs = {}
        for sensor_info in config.get("parsed_sensors", []):
            self.sensor_configs[sensor_info["name"]] = sensor_info

        # 内部状态
        self.screen_counter = {}          # 平稳度筛查计数器
        self.vibration_baseline = {}      # 振动基线（固有频率、低频能量）
        self.noise_baseline = {}          # 噪声基线
        self.level_history = {}           # 水平度历史数据 {sensor: deque of (timestamp, pitch, roll)}
        self.level_alarm_triggered = {}   # 水平度报警状态
        self.baseline_loaded = False
        self._load_baseline()

    def _get_baseline_path(self) -> str:
        import sys
        return os.path.join(sys.path[0], "elevator_car_baseline.json")

    def _load_baseline(self):
        """加载基线文件（包括振动和噪声基线）"""
        path = self._get_baseline_path()
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    baseline = json.load(f)
                self.vibration_baseline = baseline.get("vibration", {})
                self.noise_baseline = baseline.get("noise", {})
                self.baseline_loaded = True
                self.mylog.info(f"轿厢基线加载成功: {path}")
            except Exception as e:
                self.mylog.error(f"加载轿厢基线失败: {e}")

    def _get_vibration_baseline(self, sensor_name: str) -> Optional[Dict]:
        return self.vibration_baseline.get(sensor_name)

    def _get_noise_baseline(self, sensor_name: str) -> Optional[Dict]:
        return self.noise_baseline.get(sensor_name)

    # ==================== 辅助特征提取函数 ====================
    def _compute_stability_score(self, pf_dict: Dict, imp_dict: Dict, mar_dict: Dict) -> bool:
        """
        平稳度异常筛查：X/Y轴时域指标超阈值判断
        至少两个指标超阈值即认为异常
        """
        x_pf = pf_dict.get("X", 0)
        y_pf = pf_dict.get("Y", 0)
        x_imp = imp_dict.get("X", 0)
        y_imp = imp_dict.get("Y", 0)
        x_mar = mar_dict.get("X", 0)
        y_mar = mar_dict.get("Y", 0)

        # 分别对X和Y轴评分，任一轴满足条件即可
        score_x = (x_pf > self.stab_pf_thresh) + (x_imp > self.stab_if_thresh) + (x_mar > self.stab_mf_thresh)
        score_y = (y_pf > self.stab_pf_thresh) + (y_imp > self.stab_if_thresh) + (y_mar > self.stab_mf_thresh)
        return (score_x >= 2) or (score_y >= 2)

    def _compute_low_freq_energy(self, spectrum: np.ndarray, freqs: np.ndarray,
                                 low: float = 0.5, high: float = 20.0) -> float:
        """计算指定低频段能量"""
        mask = (freqs >= low) & (freqs <= high)
        return float(np.sum(spectrum[mask] ** 2))

    def _find_natural_freqs(self, spectrum: np.ndarray, freqs: np.ndarray,
                            low: float = 5, high: float = 200) -> List[float]:
        """在Z轴低频段寻找前三阶固有频率"""
        mask = (freqs >= low) & (freqs <= high)
        sub_spec = spectrum[mask]
        sub_freq = freqs[mask]
        if len(sub_spec) == 0:
            return []
        peaks, _ = find_peaks(sub_spec, height=0.2 * np.max(sub_spec), distance=5)
        if len(peaks) == 0:
            return []
        idx_sorted = sorted(peaks, key=lambda x: sub_spec[x], reverse=True)[:3]
        f_list = sub_freq[idx_sorted].tolist()
        f_list.sort()
        return f_list

    # ==================== 子故障诊断方法 ====================
    def _check_frame_vibration(self, sensor_name: str, f3_current: List[float]) -> Tuple[bool, Optional[Dict]]:
        """轿架振动异常：固有频率下降检测"""
        baseline = self._get_vibration_baseline(sensor_name)
        if not baseline:
            return False, None
        base_f3 = baseline.get("f3", [])
        if len(base_f3) < 3 or len(f3_current) < 3:
            return False, None
        shift = (base_f3[2] - f3_current[2]) / base_f3[2]
        is_fault = shift > self.freq_shift_ratio
        extra = {
            "f3_shift_ratio": shift,
            "base_f3": base_f3,
            "current_f3": f3_current
        }
        return is_fault, extra

    def _check_stability(self, sensor_name: str, fft_all: Dict) -> Tuple[bool, Optional[Dict]]:
        """平稳度异常：X/Y轴低频能量升高检测"""
        if "X" not in fft_all or "Y" not in fft_all:
            return False, None
        spec_x = np.array(fft_all["X"]["fft"])
        freq_x = np.array(fft_all["X"]["index"])
        spec_y = np.array(fft_all["Y"]["fft"])
        freq_y = np.array(fft_all["Y"]["index"])

        energy_x = self._compute_low_freq_energy(spec_x, freq_x)
        energy_y = self._compute_low_freq_energy(spec_y, freq_y)

        baseline = self._get_vibration_baseline(sensor_name)
        if not baseline:
            return False, None
        base_energy_x = baseline.get("low_energy_X", energy_x)
        base_energy_y = baseline.get("low_energy_Y", energy_y)

        ratio_x = energy_x / (base_energy_x + 1e-10)
        ratio_y = energy_y / (base_energy_y + 1e-10)
        is_fault = (ratio_x > self.stab_low_energy_ratio) or (ratio_y > self.stab_low_energy_ratio)
        extra = {
            "energy_X": energy_x,
            "base_energy_X": base_energy_x,
            "ratio_X": ratio_x,
            "energy_Y": energy_y,
            "base_energy_Y": base_energy_y,
            "ratio_Y": ratio_y
        }
        return is_fault, extra

    def _check_friction_noise(self, sensor_name: str, data_packet: Dict) -> Tuple[bool, Optional[Dict]]:
        """摩擦噪声异常：麦克风特征频段能量+振动相干性"""
        audio = data_packet.get("audio")
        if not audio:
            return False, None

        # 特征频段能量（1-5kHz），建议发送端预计算
        band_energy = audio.get("rms_band_1k_5k", 0)
        baseline = self._get_noise_baseline(sensor_name)
        if not baseline:
            return False, None
        base_energy = baseline.get("band_energy", band_energy * 0.5)
        ratio = band_energy / (base_energy + 1e-10)

        # 可选：计算声-振相干性
        coherence_ok = True
        if self.noise_coherence_thresh > 0:
            # 需要原始音频和振动信号，此处仅示意
            pass

        is_fault = (ratio > self.noise_band_ratio) and coherence_ok
        extra = {
            "band_energy": band_energy,
            "base_energy": base_energy,
            "energy_ratio": ratio
        }
        return is_fault, extra

    def _check_level_deviation(self, sensor_name: str, data_packet: Dict) -> Tuple[bool, Optional[Dict]]:
        """水平度偏差：倾角仪角度长期监测"""
        incl = data_packet.get("inclinometer")
        if not incl:
            return False, None

        pitch = incl.get("pitch", 0.0)
        roll = incl.get("roll", 0.0)
        timestamp = data_packet.get("timestamp", 0)

        if sensor_name not in self.level_history:
            self.level_history[sensor_name] = deque(maxlen=1000)
        self.level_history[sensor_name].append((timestamp, pitch, roll))

        # 计算综合倾斜角度
        tilt_angle = np.sqrt(pitch**2 + roll**2)

        # 如果当前倾斜超阈值，且持续时间超过阈值，则报警
        if tilt_angle > self.level_angle_thresh:
            if sensor_name not in self.level_alarm_triggered:
                self.level_alarm_triggered[sensor_name] = {"start_time": timestamp, "triggered": False}
            alarm_state = self.level_alarm_triggered[sensor_name]
            if not alarm_state["triggered"]:
                if timestamp - alarm_state["start_time"] >= self.level_duration_thresh:
                    alarm_state["triggered"] = True
                    return True, {"tilt_angle": tilt_angle, "pitch": pitch, "roll": roll,
                                  "duration": timestamp - alarm_state["start_time"]}
        else:
            if sensor_name in self.level_alarm_triggered:
                del self.level_alarm_triggered[sensor_name]

        return False, None

    # ==================== 主更新接口 ====================
    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        # 只处理稳态数据（若数据包提供running_state）
        if data_packet.get("running_state") not in ("steady", None):
            return False, None

        # 获取各传感器数据
        pf = data_packet.get("peak_factor", {})
        imp = data_packet.get("impulse_factor", {})
        mar = data_packet.get("margin_factor", {})
        fft_all = data_packet.get("fft_all", {})

        # 结果收集
        faults = {}
        extra_all = {}

        # ----- 1. 平稳度异常筛查（连续超阈值） -----
        if pf and imp and mar:
            abnormal = self._compute_stability_score(pf, imp, mar)
            cnt = self.screen_counter.get(sensor_name, 0)
            if abnormal:
                cnt += 1
            else:
                cnt = 0
            self.screen_counter[sensor_name] = cnt

            if cnt >= self.trigger_count and fft_all:
                # 二级确认：低频能量检测
                stab_fault, stab_extra = self._check_stability(sensor_name, fft_all)
                if stab_fault:
                    faults["stability"] = True
                    extra_all["stability"] = stab_extra

        # ----- 2. 轿架振动异常（直接频域分析，不需时域筛查）-----
        if fft_all and "Z" in fft_all:
            spec_z = np.array(fft_all["Z"]["fft"])
            freq_z = np.array(fft_all["Z"]["index"])
            f3_current = self._find_natural_freqs(spec_z, freq_z)
            frame_fault, frame_extra = self._check_frame_vibration(sensor_name, f3_current)
            if frame_fault:
                faults["frame_vibration"] = True
                extra_all["frame_vibration"] = frame_extra

        # ----- 3. 摩擦噪声异常 -----
        noise_fault, noise_extra = self._check_friction_noise(sensor_name, data_packet)
        if noise_fault:
            faults["friction_noise"] = True
            extra_all["friction_noise"] = noise_extra

        # ----- 4. 水平度偏差 -----
        level_fault, level_extra = self._check_level_deviation(sensor_name, data_packet)
        if level_fault:
            faults["level_deviation"] = True
            extra_all["level_deviation"] = level_extra

        final_fault = len(faults) > 0
        extra_all["fault_types"] = list(faults.keys())
        extra_all["sensor_name"] = sensor_name

        return final_fault, extra_all

    def reset(self, sensor_name: Optional[str] = None):
        if sensor_name:
            self.screen_counter.pop(sensor_name, None)
            self.level_history.pop(sensor_name, None)
            self.level_alarm_triggered.pop(sensor_name, None)
        else:
            self.screen_counter.clear()
            self.level_history.clear()
            self.level_alarm_triggered.clear()