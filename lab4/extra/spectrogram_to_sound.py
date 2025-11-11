import numpy as np
from PIL import Image
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def load_image_rgb(image_path: str) -> np.ndarray:
    """Load image and return an RGB float32 array in range 0..1.

    Args:
        image_path: Path to the PNG file.

    Returns:
        img_rgb: (H, W, 3) array float32 in [0,1].
    """
    img = Image.open(image_path).convert('RGB')
    img_rgb = np.array(img).astype(np.float32) / 255.0
    return img_rgb


def invert_colormap_to_magnitude(img_rgb: np.ndarray, cmap_name: str = 'plasma', n_cmap: int = 256) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Map RGB pixels to nearest colormap index and return a uint8 magnitude map.

    Returns:
        spectrogram_data: (H, W) uint8 magnitudes 0..255
        idx_map: (H, W) int indices into colormap (or None if fallback used)
        n_cmap: number of colors used
    """
    try:
        cmap = plt.get_cmap(cmap_name)(np.linspace(0, 1, n_cmap))[:, :3]
        h, w, _ = img_rgb.shape
        pixels = img_rgb.reshape(-1, 3)

        # Compute squared distances between pixels and colormap colors
        diffs = pixels[:, None, :] - cmap[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        idx = np.argmin(d2, axis=1).reshape(h, w)
        norm_mag = idx / float(n_cmap - 1)
        spectrogram_data = (norm_mag * 255.0).astype(np.uint8)
        return spectrogram_data, idx, n_cmap
    except Exception:
        # Fallback to grayscale
        img_gray = Image.fromarray((img_rgb * 255).astype(np.uint8)).convert('L')
        spectrogram_data = np.array(img_gray)
        return spectrogram_data, None, n_cmap


def create_yellow_mask(idx_map: Optional[np.ndarray], spectrogram_data: np.ndarray, n_cmap: int, threshold_pct: float = 90.0) -> np.ndarray:
    """Create a black/white mask: black = yellow/high amplitude, white = otherwise.

    If idx_map is available, marks top (100-threshold_pct)% colormap indices as yellow.
    Otherwise thresholds spectrogram_data at the threshold_pct percentile.
    """
    h, w = spectrogram_data.shape
    if idx_map is not None:
        threshold_idx = int((threshold_pct / 100.0) * (n_cmap - 1))
        mask = idx_map >= threshold_idx
    else:
        thr = np.percentile(spectrogram_data, threshold_pct)
        mask = spectrogram_data >= thr
    mask_img = np.where(mask, 0, 255).astype(np.uint8)  # black where True
    return mask_img


def magnitude_to_amplitude(spectrogram_data: np.ndarray, db_contrast: float = 80.0) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 0..255 magnitude to amplitude (linear) with dB mapping.

    Maps pixel=255 -> 0 dB and pixel=0 -> -db_contrast dB.
    Returns both amplitude (linear) and amplitude_db arrays.
    """
    amplitude_db = (spectrogram_data.astype(np.float32) / 255.0 - 1.0) * db_contrast
    amplitude = 10.0 ** (amplitude_db / 20.0)
    return amplitude, amplitude_db


def synthesize_audio(amplitude: np.ndarray, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """Synthesize a time-domain signal from magnitude spectrogram (no phase reconstruction).

    amplitude: shape (freq_bins, time_frames)
    duration: seconds
    Returns normalized float32 signal in [-1,1].
    """
    height, width = amplitude.shape
    n_fft = (height - 1) * 2
    hop_length = int(sample_rate * duration / max(1, (width - 1)))
    signal = np.zeros(int(duration * sample_rate), dtype=np.float32)

    for i in range(width):
        time_slice_amplitude = amplitude[:, i]
        spectrum = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
        max_bin = min(height, len(spectrum))
        spectrum[:max_bin] = time_slice_amplitude[:max_bin]
        time_segment = np.fft.irfft(spectrum, n=n_fft).astype(np.float32)
        window = np.hanning(len(time_segment)).astype(np.float32)
        start_sample = i * hop_length
        end_sample = start_sample + len(time_segment)
        if start_sample < len(signal):
            if end_sample <= len(signal):
                signal[start_sample:end_sample] += time_segment * window
            else:
                n_fit = len(signal) - start_sample
                signal[start_sample:] += time_segment[:n_fit] * window[:n_fit]

    # Normalize
    max_abs = np.max(np.abs(signal))
    if max_abs > 0:
        signal /= max_abs
    return signal


def save_wav(path: str, sample_rate: int, signal: np.ndarray) -> None:
    wavfile.write(path, sample_rate, signal.astype(np.float32))


def save_spectrogram_from_wav(wav_path: str, out_spec_path: str, freq_max: int = 10000, cmap: str = 'plasma') -> None:
    """Create and save a spectrogram PNG from a WAV file."""
    try:
        sr, audio = wavfile.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        plt.figure(figsize=(10, 4))
        Pxx, freqs, bins, im = plt.specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap=cmap)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.ylim(0, min(freq_max, sr / 2))
        cbar = plt.colorbar(im)
        cbar.set_label('Intensity')
        plt.savefig(out_spec_path, dpi=200, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not generate spectrogram: {e}")


def save_diagnostics(image_path: str, spectrogram_data: np.ndarray, amplitude_db: np.ndarray, mask_img: np.ndarray) -> None:
    """Save recovered magnitude, amplitude dB visualization, and mask images."""
    try:
        recovered_mag_path = image_path.replace('.png', '_recovered_mag.png')
        Image.fromarray(spectrogram_data).save(recovered_mag_path)
        ampdb_vis = ((amplitude_db - amplitude_db.min()) / (amplitude_db.ptp() + 1e-12) * 255).astype(np.uint8)
        ampdb_path = image_path.replace('.png', '_amplitude_db.png')
        Image.fromarray(ampdb_vis).save(ampdb_path)
        mask_path = image_path.replace('.png', '_yellow_mask.png')
        Image.fromarray(mask_img).save(mask_path)
        print(f"Saved diagnostics: {recovered_mag_path}, {ampdb_path}, {mask_path}")
    except Exception as e:
        print(f"Could not save diagnostics: {e}")


def image_to_spectrogram_and_audio(image_path: str, duration: float = 1.0, freq_max: int = 10000, db_contrast: float = 80.0) -> None:
    """Top-level orchestration: read image, invert colormap, synthesize audio, and save artifacts."""
    img_rgb = load_image_rgb(image_path)
    spectrogram_data, idx_map, n_cmap = invert_colormap_to_magnitude(img_rgb, cmap_name='plasma', n_cmap=256)
    mask_img = create_yellow_mask(idx_map, spectrogram_data, n_cmap, threshold_pct=90.0)
    amplitude, amplitude_db = magnitude_to_amplitude(spectrogram_data, db_contrast=db_contrast)

    # Save diagnostics before synthesis
    save_diagnostics(image_path, spectrogram_data, amplitude_db, mask_img)

    # Synthesize audio
    sample_rate = 44100
    signal = synthesize_audio(amplitude, duration, sample_rate=sample_rate)
    print(f"Signal stats: min={signal.min():.6f}, max={signal.max():.6f}, mean={signal.mean():.6f}")

    # Save WAV
    output_wav = image_path.replace('.png', '.wav')
    save_wav(output_wav, sample_rate, signal)
    print(f"Audio file saved to {output_wav}")

    # Save spectrogram PNG from the WAV for verification
    spec_path = output_wav.replace('.wav', '_spec.png')
    save_spectrogram_from_wav(output_wav, spec_path, freq_max=freq_max, cmap='plasma')
    print(f"Spectrogram image saved to {spec_path}")


if __name__ == '__main__':
    image_to_spectrogram_and_audio('lab4/extra/Screenshot from 2025-11-04 17-05-43.png')