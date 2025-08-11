"""
EDF Analyzer Class
File: edf_analyzer.py

Comprehensive EDA utilities for EDF (European Data Format) files:
- Load files + rich metadata
- Plot raw signals (per-channel sample rates)
- Amplitude distributions
- Power Spectral Density (Welch)
- Spectrogram

Author: Analysis Tools (revised, emoji-free)
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyedflib
from scipy import signal
from scipy.stats import skew, kurtosis

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


class EDFAnalyzer:
    def __init__(self, data_path, style='default'):
        self.data_path = Path(data_path)
        self.files = sorted(self.data_path.glob("*.edf"))
        self.data = {}
        self.metadata = {}

        try:
            plt.style.use(style)
        except Exception:
            plt.style.use('default')

        if _HAS_SNS:
            try:
                sns.set_palette("husl")
            except Exception:
                pass

        print(f"Initialized EDFAnalyzer with {len(self.files)} EDF files in {self.data_path}")

    def load_single_file(self, file_path, verbose=True):
        file_path = Path(file_path)
        try:
            f = pyedflib.EdfReader(str(file_path))

            n_sig = f.signals_in_file
            sample_rates = [f.getSampleFrequency(i) for i in range(n_sig)]
            labels = f.getSignalLabels()
            phys_dims = [f.getPhysicalDimension(i) for i in range(n_sig)]
            phys_mins = [f.getPhysicalMinimum(i) for i in range(n_sig)]
            phys_maxs = [f.getPhysicalMaximum(i) for i in range(n_sig)]

            signals = [f.readSignal(i) for i in range(n_sig)]
            duration = f.file_duration
            start_time = f.getStartdatetime()

            f.close()

            metadata = {
                "filename": file_path.name,
                "file_path": str(file_path),
                "n_signals": n_sig,
                "signal_labels": labels,
                "sample_rates": sample_rates,
                "physical_dimensions": phys_dims,
                "physical_mins": phys_mins,
                "physical_maxs": phys_maxs,
                "duration": duration,
                "start_time": start_time,
            }

            if verbose:
                fs_repr = "mixed" if len(set(sample_rates)) > 1 else f"{sample_rates[0]} Hz"
                print(f"Loaded {file_path.name} — {n_sig} channels, {duration:.1f}s, fs={fs_repr}")

            return np.asarray(signals, dtype=object), metadata

        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return None, None

    def load_all_files(self, verbose=True):
        if not self.files:
            print("No EDF files found in the specified directory.")
            return

        print(f"Loading {len(self.files)} EDF files...\n" + "-" * 60)
        ok = 0
        for fp in self.files:
            signals, meta = self.load_single_file(fp, verbose=verbose)
            if signals is not None:
                key = fp.stem
                self.data[key] = signals
                self.metadata[key] = meta
                ok += 1
        print("-" * 60)
        print(f"Successfully loaded {ok}/{len(self.files)} files")

    def get_file_list(self):
        return list(self.data.keys())

    def get_metadata_summary(self, filename=None):
        if filename and filename in self.metadata:
            return self.metadata[filename]

        rows = []
        for key, meta in self.metadata.items():
            sr = "mixed"
            if meta.get("sample_rates"):
                sr = meta["sample_rates"][0] if len(set(meta["sample_rates"])) == 1 else "mixed"
            rows.append({
                "File": meta.get("filename", key),
                "Channels": meta.get("n_signals", None),
                "Duration_s": meta.get("duration", None),
                "Sample_Rate": sr,
                "Start_Time": meta.get("start_time", None),
            })
        return pd.DataFrame(rows)

    def plot_raw_signals(self, filename=None, max_duration=30, max_channels=8,
                         figsize=(15, 12), show_grid=True, detrend=False):
        if not self.data:
            print("No data loaded. Call load_all_files() first.")
            return
        if filename is None:
            filename = next(iter(self.data))

        if filename not in self.data:
            print(f"File {filename} not found")
            return

        sigs = self.data[filename]
        meta = self.metadata[filename]
        n_ch = min(len(sigs), max_channels)

        fig, axes = plt.subplots(n_ch, 1, figsize=figsize, sharex=False)
        if n_ch == 1:
            axes = [axes]

        for i in range(n_ch):
            fs = float(meta["sample_rates"][i])
            N = int(max_duration * fs)
            y = np.asarray(sigs[i][:N])
            if detrend and y.size:
                y = y - np.mean(y)
            t = np.arange(len(y)) / fs

            ax = axes[i]
            ax.plot(t, y, linewidth=0.9, alpha=0.9)

            label = meta["signal_labels"][i] if i < len(meta["signal_labels"]) else f"Ch {i+1}"
            unit = meta["physical_dimensions"][i] if i < len(meta["physical_dimensions"]) else ""
            ax.set_ylabel(f"{label}\n({unit})", fontsize=10)
            if show_grid:
                ax.grid(True, alpha=0.3)

            ax.text(0.02, 0.95, f"μ={np.mean(y):.2f}, σ={np.std(y):.2f}",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        axes[-1].set_xlabel("Time (s)", fontsize=12)
        plt.suptitle(f"Raw Signals — {meta['filename']} (First {max_duration}s)", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

    def calculate_signal_statistics(self, filename=None, return_df=True):
        if not self.data:
            print("No data loaded.")
            return None
        if filename is None:
            filename = next(iter(self.data))
        if filename not in self.data:
            print(f"File {filename} not found")
            return None

        sigs = self.data[filename]
        meta = self.metadata[filename]

        rows = []
        for i, y in enumerate(sigs):
            y = np.asarray(y)
            rows.append({
                "Channel": meta["signal_labels"][i] if i < len(meta["signal_labels"]) else f"Ch_{i+1}",
                "Mean": float(np.mean(y)),
                "Std": float(np.std(y)),
                "Median": float(np.median(y)),
                "Min": float(np.min(y)),
                "Max": float(np.max(y)),
                "Range": float(np.ptp(y)),
                "RMS": float(np.sqrt(np.mean(y ** 2))),
                "Variance": float(np.var(y)),
                "Skewness": float(skew(y)),
                "Kurtosis": float(kurtosis(y)),
                "Zero_Crossings": int(np.count_nonzero(np.diff(np.sign(y)))),
                "Sample_Count": int(len(y)),
            })
        df = pd.DataFrame(rows)
        if return_df:
            return df
        print(df.round(4))
        return df

    def plot_amplitude_distributions(self, filename=None, max_channels=8, bins=50, figsize=(16, 10)):
        if not self.data:
            print("No data loaded.")
            return
        if filename is None:
            filename = next(iter(self.data))
        if filename not in self.data:
            print(f"File {filename} not found")
            return

        sigs = self.data[filename]
        meta = self.metadata[filename]
        n_ch = min(len(sigs), max_channels)

        cols = 3
        rows = (n_ch + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).reshape(rows, cols).flatten()

        for i in range(n_ch):
            y = np.asarray(sigs[i])
            ax = axes[i]
            ax.hist(y, bins=bins, alpha=0.7, density=True, edgecolor="black", linewidth=0.5)
            mean_val, std_val = np.mean(y), np.std(y)
            ax.axvline(mean_val, linestyle="--", alpha=0.8)
            ax.axvline(mean_val + std_val, linestyle=":", alpha=0.8)
            ax.axvline(mean_val - std_val, linestyle=":", alpha=0.8)

            label = meta["signal_labels"][i] if i < len(meta["signal_labels"]) else f"Ch {i+1}"
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Amplitude", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.grid(True, alpha=0.3)

        for j in range(n_ch, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"Amplitude Distributions — {meta['filename']}", fontsize=14)
        plt.tight_layout()
        plt.show()

    def power_spectral_analysis(self, filename=None, max_channels=6, max_freq=100,
                                nperseg=2048, figsize=(16, 12)):
        if not self.data:
            print("No data loaded.")
            return
        if filename is None:
            filename = next(iter(self.data))
        if filename not in self.data:
            print(f"File {filename} not found")
            return

        sigs = self.data[filename]
        meta = self.metadata[filename]
        n_ch = min(len(sigs), max_channels)

        cols = 3
        rows = (n_ch + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).reshape(rows, cols).flatten()

        psd_results = {}

        for i in range(n_ch):
            y = np.asarray(sigs[i])
            fs = float(meta["sample_rates"][i])
            seg = max(64, min(len(y) // 4, nperseg))
            freqs, psd = signal.welch(y, fs=fs, nperseg=seg)

            mask = freqs <= max_freq
            ax = axes[i]
            ax.semilogy(freqs[mask], psd[mask], linewidth=1.2)
            label = meta["signal_labels"][i] if i < len(meta["signal_labels"]) else f"Ch {i+1}"
            ax.set_title(f"PSD — {label}", fontsize=10)
            ax.set_xlabel("Frequency (Hz)", fontsize=9)
            ax.set_ylabel("PSD (unit²/Hz)", fontsize=9)
            ax.grid(True, alpha=0.3)

            psd_results[label] = {"freqs": freqs, "psd": psd, "fs": fs}

        for j in range(n_ch, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"Power Spectral Density — {meta['filename']}", fontsize=14)
        plt.tight_layout()
        plt.show()

        return psd_results

    def create_spectrogram(self, filename=None, channel_idx=0, max_duration=120,
                           nperseg=256, max_freq=50, figsize=(14, 8)):
        if not self.data:
            print("No data loaded.")
            return
        if filename is None:
            filename = next(iter(self.data))
        if filename not in self.data:
            print(f"File {filename} not found")
            return

        sigs = self.data[filename]
        meta = self.metadata[filename]

        if channel_idx >= len(sigs):
            print(f"Channel index {channel_idx} out of range (0..{len(sigs)-1})")
            return

        fs = float(meta["sample_rates"][channel_idx])
        N = int(max_duration * fs)
        y = np.asarray(sigs[channel_idx][:N])

        f, t, Sxx = signal.spectrogram(y, fs=fs, nperseg=max(64, min(N, nperseg)))
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        plt.figure(figsize=figsize)
        plt.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="viridis")
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.xlabel("Time (s)", fontsize=12)
        ch_name = meta["signal_labels"][channel_idx] if channel_idx < len(meta["signal_labels"]) else f"Ch {channel_idx+1}"
        plt.title(f"Spectrogram — {ch_name} ({meta['filename']})", fontsize=14)
        plt.colorbar(label="Power (dB)", shrink=0.85)
        plt.ylim([0, min(max_freq, fs/2)])
        plt.tight_layout()
        plt.show()

        return f, t, Sxx
