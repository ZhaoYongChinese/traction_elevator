import os
import json
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from scipy.signal import find_peaks
from .base import BaseFaultDetector


class RopeFaultDetector(BaseFaultDetector):
    """
    钢丝绳故障检测器（伸长/松动 + 张力不均）
    采用两级诊断：
    1. 时域指标（峰值因子/脉冲因子/裕度因子）连续超阈值筛查
    2. 频域精细诊断：固有频率下降检测伸长，绳通过频率边带检测张力不均
    支持载重补偿，可存储多载重级别的基线。
    """

    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        super().__init__(name, config)
        # 通用参数
        self.window_size = self.params.get("window_size", 10)
        self.trigger_count = self.params.get("trigger_count", 3)

        # 时域筛查阈值
        self.pf_thresh = self.params.get("peak_factor_thresh", 5.0)
        self.if_thresh = self.params.get("impulse_factor_thresh", 6.0)
        self.mf_thresh = self.params.get("margin_factor_thresh", 7.0)

        # 频域诊断阈值
        self.freq_shift_ratio = self.params.get("freq_shift_ratio", 0.05)
        self.fp_amp_ratio = self.params.get("rope_pass_amp_ratio", 3.0)
        self.sideband_check = self.params.get("sideband_check", True)

        # 载重补偿
        self.load_compensation = self.params.get("load_compensation", False)
        self.load_levels = self.params.get("load_levels", [])

        # 传感器独立配置
        self.sensor_configs = {}
        for sensor_info in config.get("parsed_sensors", []):
            sensor_name = sensor_info["name"]
            self.sensor_configs[sensor_name] = sensor_info

        # 内部状态
        self.screen_counter = {}      # 一级筛查计数器

        # 基线数据结构：{sensor_name: {"load_XXX": {"f3": [...], "fp_amp": float}}}
        self.baseline = {}
        self.baseline_loaded = False
        self._load_baseline()

    def _get_baseline_path(self) -> str:
        """获取基线文件路径"""
        import sys
        base_dir = sys.path[0]
        return os.path.join(base_dir, "rope_baseline.json")

    def _load_baseline(self):
        """从 JSON 文件加载健康基线"""
        path = self._get_baseline_path()
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.baseline = json.load(f)
                self.baseline_loaded = True
                self.mylog.info(f"钢丝绳基线加载成功: {path}")
            except Exception as e:
                self.mylog.error(f"加载基线文件失败: {e}")
                self.baseline_loaded = False
        else:
            self.mylog.warning(f"基线文件不存在: {path}，请先运行标定脚本")

    def _get_baseline_for_load(self, sensor_name: str, load_weight: Optional[float]) -> Optional[Dict]:
        """根据载重获取最匹配的基线"""
        if sensor_name not in self.baseline:
            return None
        sensor_base = self.baseline[sensor_name]
        if not self.load_compensation or load_weight is None or not self.load_levels:
            # 无载重补偿或载重未知，返回第一个基线（或名为 "default" 的）
            return sensor_base.get("default") or next(iter(sensor_base.values()), None)

        # 找到最接近的载重级别
        closest_load = min(self.load_levels, key=lambda x: abs(x - load_weight))
        key = f"load_{int(closest_load)}"
        return sensor_base.get(key)

    def _compute_time_features(self, pf_dict: Dict, imp_dict: Dict, mar_dict: Dict) -> bool:
        """
        时域特征联合判断，返回是否异常
        策略：至少两个指标超阈值即认为异常
        """
        z_pf = pf_dict.get("Z", 0)
        z_imp = imp_dict.get("Z", 0)
        z_mar = mar_dict.get("Z", 0)
        score = (z_pf > self.pf_thresh) + (z_imp > self.if_thresh) + (z_mar > self.mf_thresh)
        return score >= 2

    def _find_natural_freqs(self, spectrum: np.ndarray, freqs: np.ndarray,
                            low: float = 5, high: float = 200) -> List[float]:
        """在低频段寻找前三阶固有频率峰值"""
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

    def _compute_band_energy(self, spectrum: np.ndarray, freqs: np.ndarray, center_freq: float, bandwidth: float = 2.0) -> float:
        """计算中心频率附近带宽内的能量"""
        mask = (freqs >= center_freq - bandwidth) & (freqs <= center_freq + bandwidth)
        return float(np.sum(spectrum[mask] ** 2))

    def _check_elongation(self, sensor_name: str, f3_current: List[float],
                          spec_z: np.ndarray, freq_z: np.ndarray, baseline: Dict) -> Tuple[bool, Optional[Dict]]:
        """检查固有频率是否下降（使用频段能量比较，更稳健）"""
        if not baseline:
            return False, None
        base_f3 = baseline.get("f3", [])
        if len(base_f3) < 3 or len(f3_current) < 3:
            return False, None

        # 方法1：峰值频率偏移
        shift_peak = (base_f3[2] - f3_current[2]) / base_f3[2] if base_f3[2] != 0 else 0

        # 方法2：频段能量变化（推荐）
        base_energy = baseline.get("f3_energy")
        if base_energy is None and base_f3:
            # 若基线未存储能量，动态计算
            base_energy = self._compute_band_energy(spec_z, freq_z, base_f3[2])
        current_energy = self._compute_band_energy(spec_z, freq_z, f3_current[2])
        energy_ratio = current_energy / (base_energy + 1e-10)
        # 能量下降超过阈值（能量比<1表示下降）
        energy_drop = energy_ratio < (1 - self.freq_shift_ratio)

        is_fault = shift_peak > self.freq_shift_ratio or energy_drop
        extra = {
            "f3_shift_ratio": shift_peak,
            "base_f3": base_f3,
            "current_f3": f3_current,
            "base_energy": base_energy,
            "current_energy": current_energy,
            "energy_ratio": energy_ratio
        }
        return is_fault, extra

    def _check_tension_imbalance(self, sensor_name: str, spectrum: np.ndarray, freqs: np.ndarray,
                                 rope_speed: float, lay_length: float, sheave_dia: float,
                                 baseline: Dict) -> Tuple[bool, Optional[Dict]]:
        """检查张力不均：fₚ幅值升高及边带"""
        if rope_speed is None:
            return False, None
        # 计算理论特征频率
        fp = rope_speed / lay_length
        fr = rope_speed / (np.pi * sheave_dia)

        idx_fp = np.argmin(np.abs(freqs - fp))
        fp_amp = spectrum[idx_fp]

        base_fp_amp = baseline.get("fp_amp") if baseline else None
        if base_fp_amp is None:
            base_fp_amp = fp_amp * 0.5

        amp_ratio = fp_amp / (base_fp_amp + 1e-10)

        has_sideband = False
        if self.sideband_check:
            for n in [1, 2]:
                for sign in [-1, 1]:
                    target = fp + sign * n * fr
                    idx = np.argmin(np.abs(freqs - target))
                    if 0 < idx < len(spectrum) - 1:
                        local_mean = np.mean(spectrum[max(0, idx-3):min(len(spectrum), idx+4)])
                        if spectrum[idx] > local_mean * 3:
                            has_sideband = True
                            break
                if has_sideband:
                    break

        is_fault = (amp_ratio > self.fp_amp_ratio) and has_sideband
        extra = {
            "fp_theory": fp,
            "fr_theory": fr,
            "fp_amp": float(fp_amp),
            "base_fp_amp": float(base_fp_amp),
            "amp_ratio": float(amp_ratio),
            "has_sideband": has_sideband
        }
        return is_fault, extra

    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        # 仅处理稳态数据
        if data_packet.get("running_state") != "steady":
            return False, None

        pf = data_packet.get("peak_factor", {})
        imp = data_packet.get("impulse_factor", {})
        mar = data_packet.get("margin_factor", {})
        fft_all = data_packet.get("fft_all", {})
        load_weight = data_packet.get("load_weight")

        # 一级筛查：时域指标
        time_abnormal = self._compute_time_features(pf, imp, mar)
        cnt = self.screen_counter.get(sensor_name, 0)
        if time_abnormal:
            cnt += 1
        else:
            cnt = 0
        self.screen_counter[sensor_name] = cnt

        is_screen_trigger = cnt >= self.trigger_count
        extra_info = {
            "fault_type": self.name,
            "screen_counter": cnt,
            "time_abnormal": time_abnormal
        }

        final_fault = False
        if is_screen_trigger and fft_all:
            if "Z" not in fft_all:
                self.mylog.warning(f"传感器 {sensor_name} 缺少Z轴频谱数据")
                return False, extra_info
            spec_z = np.array(fft_all["Z"]["fft"])
            freq_z = np.array(fft_all["Z"]["index"])

            # 获取适配载重的基线
            baseline = self._get_baseline_for_load(sensor_name, load_weight)
            if baseline is None:
                self.mylog.warning(f"传感器 {sensor_name} 无可用基线")

            # 伸长/松动检测
            f3_current = self._find_natural_freqs(spec_z, freq_z)
            if baseline:
                elong_fault, elong_extra = self._check_elongation(sensor_name, f3_current, spec_z, freq_z, baseline)
                if elong_fault:
                    final_fault = True
                    extra_info["elongation"] = elong_extra

            # 张力不均检测
            sensor_cfg = self.sensor_configs.get(sensor_name, {})
            rope_speed = data_packet.get("rope_speed") or sensor_cfg.get("default_rope_speed")
            lay_length = sensor_cfg.get("rope_lay_length", 0.12)
            sheave_dia = sensor_cfg.get("sheave_diameter", 0.6)

            if rope_speed is not None and baseline:
                tens_fault, tens_extra = self._check_tension_imbalance(
                    sensor_name, spec_z, freq_z, rope_speed, lay_length, sheave_dia, baseline
                )
                if tens_fault:
                    final_fault = True
                    extra_info["tension_imbalance"] = tens_extra

        return final_fault, extra_info

    def reset(self, sensor_name: Optional[str] = None):
        if sensor_name:
            self.screen_counter.pop(sensor_name, None)
        else:
            self.screen_counter.clear()