#!/usr/bin/env python3
"""
钢丝绳健康基线标定脚本（支持多载重级别）
运行方式：python calibrate_rope_baseline.py
"""
import sys
import os
import json
import time
import numpy as np
import zmq
from scipy.signal import find_peaks

def find_natural_freqs(spectrum, freqs, low=5, high=200):
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

def compute_band_energy(spectrum, freqs, center_freq, bandwidth=2.0):
    mask = (freqs >= center_freq - bandwidth) & (freqs <= center_freq + bandwidth)
    return float(np.sum(spectrum[mask] ** 2))

def get_fp_amplitude(spectrum, freqs, rope_speed, lay_length=0.12):
    fp = rope_speed / lay_length
    idx = np.argmin(np.abs(freqs - fp))
    return float(spectrum[idx])

def main():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://localhost:33333")  # 根据实际修改

    sensors_to_calibrate = ["WXT02_p004_9003", "WXT03_p005_9004"]
    load_levels = [0, 500, 1000]  # kg，与配置文件一致
    frames_needed = 10

    print("=== 钢丝绳基线标定工具 ===")
    print("请根据提示，调整电梯载重至对应级别，并确保电梯匀速运行。")

    baseline = {}

    for load in load_levels:
        print(f"\n>>> 请将电梯载重调整至约 {load} kg，然后按 Enter 继续...")
        input()
        print(f"开始采集载重 {load} kg 的基线数据...")

        collected = {s: [] for s in sensors_to_calibrate}
        while True:
            try:
                data = socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.1)
                continue

            sensor = data.get("sensor_name")
            if sensor not in sensors_to_calibrate:
                continue
            if data.get("running_state") != "steady":
                continue

            fft_all = data.get("fft_all", {})
            if "Z" not in fft_all:
                continue

            spec_z = np.array(fft_all["Z"]["fft"])
            freq_z = np.array(fft_all["Z"]["index"])

            f3 = find_natural_freqs(spec_z, freq_z)
            rope_speed = data.get("rope_speed")
            if rope_speed is None:
                rope_speed = 2.5
            fp_amp = get_fp_amplitude(spec_z, freq_z, rope_speed)

            if len(f3) >= 3:
                f3_energy = compute_band_energy(spec_z, freq_z, f3[2])
            else:
                f3_energy = 0.0

            collected[sensor].append((f3, fp_amp, f3_energy))
            done_sensors = [s for s, v in collected.items() if len(v) >= frames_needed]
            print(f"\r已采集: {', '.join([f'{s}: {len(collected[s])}/{frames_needed}' for s in sensors_to_calibrate])}", end="")

            if all(len(v) >= frames_needed for v in collected.values()):
                break

        print("\n处理数据...")
        for sensor, data_list in collected.items():
            if sensor not in baseline:
                baseline[sensor] = {}
            key = f"load_{load}"
            f3_sum = np.zeros(3)
            fp_amp_sum = 0.0
            f3_energy_sum = 0.0
            valid_count = 0
            for f3, fp_amp, f3_energy in data_list:
                if len(f3) >= 3:
                    f3_sum += np.array(f3[:3])
                    f3_energy_sum += f3_energy
                    valid_count += 1
                fp_amp_sum += fp_amp
            if valid_count > 0:
                avg_f3 = (f3_sum / valid_count).tolist()
                avg_f3_energy = f3_energy_sum / valid_count
            else:
                avg_f3 = []
                avg_f3_energy = 0.0
            avg_fp_amp = fp_amp_sum / len(data_list)
            baseline[sensor][key] = {
                "f3": avg_f3,
                "f3_energy": avg_f3_energy,
                "fp_amp": avg_fp_amp
            }
            print(f"传感器 {sensor} 载重 {load}kg 基线: f3={avg_f3}, energy={avg_f3_energy:.4f}, fp_amp={avg_fp_amp:.4f}")

    output_path = os.path.join(sys.path[0], "rope_baseline.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"\n基线已保存至: {output_path}")

if __name__ == "__main__":
    main()