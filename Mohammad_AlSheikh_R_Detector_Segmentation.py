import os
import numpy as np
import wfdb
import pandas as pd
from scipy.signal import butter, filtfilt
# -------- Parameters --------
TARGET_LEADS = ["MLIII", "MLII", "D3", "D2", "III", "II", "AVF"]
ANNOT_FILTER = ["s", "T", "N"]
OUTPUT_FOLDER = "IncartBeats"
WINDOW = (-0.2, 0.6)  # seconds before and after R peak
# -------- Pan–Tompkins Implementation --------
def bandpass_filter(signal, fs, lowcut=0.5, highcut=50, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)
def derivative(signal, fs):
    return np.gradient(signal)
def moving_window_integration(signal, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode="same")
def pan_tompkins_detect(signal, fs):
    # Bandpass filter
    filtered = bandpass_filter(signal, fs)
    # Differentiate
    diff = derivative(filtered, fs)
    # Square
    squared = diff**2
    # Moving window integration (~150 ms window)
    window_size = int(0.15 * fs)
    integrated = moving_window_integration(squared, window_size)
    # Thresholding
    threshold = np.mean(integrated) * 1.2
    r_peaks = np.where(integrated > threshold)[0]
    # Refine: keep local maxima only
    refined = []
    refractory = int(0.2 * fs)  # 200 ms
    last_peak = -refractory
    for idx in r_peaks:
        if idx - last_peak > refractory:
            window = signal[max(0, idx - 10): idx + 10]
            if len(window) > 0:
                peak = idx - 10 + np.argmax(window)
                refined.append(peak)
                last_peak = peak
    return np.array(refined)
# -------- Utilities --------
def get_lead_indices(record, target_leads):
    """Find indices of target leads in WFDB record (case-insensitive)."""
    lead_indices = {}
    for i, name in enumerate(record.sig_name):
        for target in target_leads:
            if name.lower() == target.lower():
                lead_indices[target.upper()] = i
    return lead_indices
def slice_beat(signal, r_idx, fs, window=WINDOW):
    """Slice one beat around R-peak with a time axis."""
    start = int(r_idx + window[0] * fs)
    end = int(r_idx + window[1] * fs)
    if start < 0 or end > len(signal):
        return None, None
    beat = signal[start:end]
    time = np.linspace(window[0], window[1], len(beat))
    return time, beat
def process_record(folder, record_name, out_folder=OUTPUT_FOLDER, window=WINDOW):
    """Process one ECG record: detect R-peaks, slice beats, save CSVs."""
    # ---- Load record ----
    record = wfdb.rdrecord(os.path.join(folder, record_name))
    fs = record.fs
    # ---- Detect R-peaks (Pan–Tompkins) ----
    sig0 = record.p_signal[:, 0]  # first channel for detection
    r_peaks = pan_tompkins_detect(sig0, fs)
    # ---- Load annotations ----
    ann = wfdb.rdann(os.path.join(folder, record_name), "atr")
    ann_samples = ann.sample
    ann_symbols = ann.symbol
    # ---- Map leads ----
    lead_indices = get_lead_indices(record, TARGET_LEADS)
    if not lead_indices:
        print(f"⚠ No target leads found in {record_name}")
        return
    # ---- Match annotations to R-peaks ----
    for i, (s, sym) in enumerate(zip(ann_samples, ann_symbols)):
        if sym not in ANNOT_FILTER:
            continue
        # Find nearest detected R-peak
        if len(r_peaks) == 0:
            continue
        r_idx = min(r_peaks, key=lambda r: abs(r - s))
        for lead, idx in lead_indices.items():
            sig = record.p_signal[:, idx]
            t, beat = slice_beat(sig, r_idx, fs, window)
            if beat is None:
                continue
            # ---- Save CSV ----
            lead_folder = os.path.join(out_folder, lead)
            os.makedirs(lead_folder, exist_ok=True)
            filename = f"{record_name}_beat{i:03d}_{sym}.csv"
            filepath = os.path.join(lead_folder, filename)
            df = pd.DataFrame({"time (s)": t, f"{lead} (mV)": beat})
            df.to_csv(filepath, index=False)
def process_folder(folder, max_records=None):
    """Process multiple records from a folder."""
    records = [f[:-4] for f in os.listdir(folder) if f.endswith(".hea")]
    records = sorted(records)
    if max_records:
        records = records[:max_records]
    for rec in records:
        print(f"\n▶ Processing {rec}...")
        process_record(folder, rec)
    print("\n✅ Done!")
# -------- Example usage --------
if __name__ == "__main__":
    folder = r""  # change to your dataset path
    process_folder(folder, max_records=)  # process first N records
