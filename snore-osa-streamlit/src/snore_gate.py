"""
Snore detection gate module.
Simple energy-based snore detection to filter out non-snoring segments.
"""
import numpy as np
from scipy import signal

def snore_prob(audio_segment: np.ndarray, sr: int = 16000) -> float:
    """
    Estimate probability that audio segment contains snoring.
    
    Args:
        audio_segment: Audio data as numpy array
        sr: Sample rate (default 16000)
        
    Returns:
        Float between 0-1 representing snore probability
    """
    if len(audio_segment) == 0:
        return 0.0
    
    # Normalize audio
    audio_segment = audio_segment.astype(np.float32)
    if np.max(np.abs(audio_segment)) > 0:
        audio_segment = audio_segment / np.max(np.abs(audio_segment))
    
    # Basic energy-based features for snore detection
    
    # 1. RMS Energy
    rms_energy = np.sqrt(np.mean(audio_segment**2))
    
    # 2. Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(audio_segment)) != 0)
    zcr = zero_crossings / len(audio_segment)
    
    # 3. Spectral features
    f, psd = signal.welch(audio_segment, fs=sr, nperseg=min(1024, len(audio_segment)//4))
    
    # Energy in snoring frequency bands (typically 20-300 Hz)
    low_freq_mask = (f >= 20) & (f <= 300)
    if np.any(low_freq_mask):
        low_freq_energy = np.sum(psd[low_freq_mask])
        total_energy = np.sum(psd)
        spectral_ratio = low_freq_energy / (total_energy + 1e-10)
    else:
        spectral_ratio = 0.0
    
    # 4. Peak frequency in snoring range
    if np.any(low_freq_mask):
        peak_freq_idx = np.argmax(psd[low_freq_mask])
        peak_freq = f[low_freq_mask][peak_freq_idx]
        # Snoring typically peaks around 40-120 Hz
        freq_score = 1.0 if 40 <= peak_freq <= 120 else 0.5
    else:
        freq_score = 0.0
    
    # Combine features into snore probability
    # This is a simple heuristic - could be replaced with a trained model
    energy_score = min(1.0, rms_energy * 10)  # Scale RMS energy
    zcr_score = 1.0 - min(1.0, zcr * 100)     # Lower ZCR = more periodic = more likely snore
    
    # Weighted combination
    snore_probability = (
        0.3 * energy_score +
        0.2 * zcr_score + 
        0.3 * spectral_ratio +
        0.2 * freq_score
    )
    
    return np.clip(snore_probability, 0.0, 1.0)
