"""
Lab 5 - Fourier Transform Analysis of Traffic Signal
Analyzing hourly traffic data to identify periodic components and DC offset
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_traffic_data(filename='Train.csv'):
    """
    Load traffic data from CSV file.
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file
        
    Returns:
    --------
    signal : ndarray
        Traffic count signal (number of cars)
    dates : list
        List of datetime objects for each sample
    """
    # Load the CSV file, skip header, and get only the Count column
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, 
                         usecols=2, dtype=float)
    
    # Load datetime information
    dates_raw = np.genfromtxt(filename, delimiter=',', skip_header=1, 
                              usecols=1, dtype=str)
    dates = [datetime.strptime(d, '%d-%m-%Y %H:%M') for d in dates_raw]
    
    return data, dates


# ============================================================================
# SIGNAL ANALYSIS FUNCTIONS
# ============================================================================

def get_sampling_info(signal, sampling_period_hours=1):
    """
    Get sampling information for the signal.
    
    Parameters:
    -----------
    signal : ndarray
        Input signal
    sampling_period_hours : float
        Sampling period in hours
        
    Returns:
    --------
    dict with keys:
        - Fs: Sampling frequency (samples per hour)
        - N: Number of samples
        - T: Total time duration in hours
        - Ts: Sampling period in hours
        - fmax: Maximum frequency (Nyquist frequency)
    """
    N = len(signal)
    Ts = sampling_period_hours  # Sampling period in hours
    Fs = 1 / Ts  # Sampling frequency: 1 sample/hour
    T = N * Ts  # Total time in hours
    fmax = Fs / 2  # Nyquist frequency
    
    return {
        'Fs': Fs,
        'N': N,
        'T': T,
        'Ts': Ts,
        'fmax': fmax
    }


def compute_fft(signal, Fs):
    """
    Compute FFT of signal and return magnitude spectrum.
    
    Parameters:
    -----------
    signal : ndarray
        Input signal
    Fs : float
        Sampling frequency
        
    Returns:
    --------
    freqs : ndarray
        Frequency vector (only positive frequencies)
    magnitude : ndarray
        Magnitude spectrum (normalized)
    X_full : ndarray
        Full FFT result (complex)
    """
    N = len(signal)
    
    # Compute FFT
    X_full = np.fft.fft(signal)
    
    # Compute magnitude and normalize
    magnitude = np.abs(X_full) / N
    
    # Use only positive frequencies (first half of spectrum)
    magnitude = magnitude[:N//2]
    
    # Generate frequency vector
    freqs = Fs * np.linspace(0, N//2, N//2) / N
    
    return freqs, magnitude, X_full


def has_dc_component(magnitude, threshold=0.1):
    """
    Check if signal has a DC component.
    
    Parameters:
    -----------
    magnitude : ndarray
        Magnitude spectrum
    threshold : float
        Threshold for DC component detection
        
    Returns:
    --------
    bool
        True if DC component is present
    float
        DC component value
    """
    dc_value = magnitude[0]
    has_dc = dc_value > threshold
    return has_dc, dc_value


def remove_dc_component(signal):
    """
    Remove DC component from signal by subtracting the mean.
    
    Parameters:
    -----------
    signal : ndarray
        Input signal
        
    Returns:
    --------
    ndarray
        Signal with DC component removed
    float
        DC component (mean value)
    """
    dc_component = np.mean(signal)
    signal_no_dc = signal - dc_component
    return signal_no_dc, dc_component


def find_top_frequencies(freqs, magnitude, n_top=4):
    """
    Find the top N frequencies with highest magnitude.
    
    Parameters:
    -----------
    freqs : ndarray
        Frequency vector
    magnitude : ndarray
        Magnitude spectrum
    n_top : int
        Number of top frequencies to find
        
    Returns:
    --------
    list of tuples
        Each tuple contains (frequency, magnitude, period_hours, period_days)
    """
    # Skip DC component (index 0)
    magnitude_no_dc = magnitude[1:]
    freqs_no_dc = freqs[1:]
    
    # Find indices of top N magnitudes
    top_indices = np.argsort(magnitude_no_dc)[-n_top:][::-1]
    
    results = []
    for idx in top_indices:
        freq = freqs_no_dc[idx]
        mag = magnitude_no_dc[idx]
        period_hours = 1 / freq if freq > 0 else np.inf
        period_days = period_hours / 24
        results.append((freq, mag, period_hours, period_days))
    
    return results


def filter_high_frequencies(signal, Fs, cutoff_freq):
    """
    Remove high frequency components from signal.
    
    Parameters:
    -----------
    signal : ndarray
        Input signal
    Fs : float
        Sampling frequency
    cutoff_freq : float
        Cutoff frequency (Hz) - frequencies above this will be removed
        
    Returns:
    --------
    ndarray
        Filtered signal
    """
    N = len(signal)
    
    # Compute FFT
    X = np.fft.fft(signal)
    
    # Create frequency vector
    freqs = np.fft.fftfreq(N, 1/Fs)
    
    # Zero out frequencies above cutoff
    X[np.abs(freqs) > cutoff_freq] = 0
    
    # Inverse FFT to get filtered signal
    filtered_signal = np.real(np.fft.ifft(X))
    
    return filtered_signal


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_full_signal(signal, title='Traffic Signal - Full Dataset'):
    """Plot the complete signal."""
    plt.figure(figsize=(12, 5))
    plt.plot(signal, linewidth=0.5)
    plt.xlabel('Samples')
    plt.ylabel('Number of Cars')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_signal_segment(signal, start_idx, duration_samples, dates=None,
                        title='Traffic Signal - One Month'):
    """
    Plot a segment of the signal.
    
    Parameters:
    -----------
    signal : ndarray
        Input signal
    start_idx : int
        Starting index
    duration_samples : int
        Number of samples to plot
    dates : list, optional
        List of datetime objects
    title : str
        Plot title
    """
    end_idx = start_idx + duration_samples
    segment = signal[start_idx:end_idx]
    
    plt.figure(figsize=(14, 5))
    
    if dates is not None:
        date_segment = dates[start_idx:end_idx]
        plt.plot(date_segment, segment, linewidth=1)
        plt.xlabel('Date')
        plt.xticks(rotation=45)
    else:
        plt.plot(range(len(segment)), segment, linewidth=1)
        plt.xlabel('Samples')
    
    plt.ylabel('Number of Cars')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fft_spectrum(freqs, magnitude, title='FFT Magnitude Spectrum', 
                      highlight_peaks=None):
    """
    Plot FFT magnitude spectrum.
    
    Parameters:
    -----------
    freqs : ndarray
        Frequency vector
    magnitude : ndarray
        Magnitude spectrum
    title : str
        Plot title
    highlight_peaks : list of tuples, optional
        List of (frequency, magnitude, period_hours, period_days) to highlight
    """
    plt.figure(figsize=(12, 5))
    plt.plot(freqs, magnitude, linewidth=1, label='Spectrum')
    
    # Highlight peaks if provided
    if highlight_peaks is not None:
        peak_freqs = [p[0] for p in highlight_peaks]
        peak_mags = [p[1] for p in highlight_peaks]
        
        plt.scatter(peak_freqs, peak_mags, color='red', s=100, 
                   zorder=5, label='Top 4 Peaks', marker='o')
        
        # Annotate each peak
        for i, (freq, mag, period_h, period_d) in enumerate(highlight_peaks, 1):
            # Format the label based on period
            if period_d >= 1:
                label = f'{i}. {period_d:.1f}d'
            else:
                label = f'{i}. {period_h:.1f}h'
            
            plt.annotate(label, 
                        xy=(freq, mag), 
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Frequency (cycles/hour)')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison(original, filtered, start_idx=0, duration_samples=None,
                   title='Original vs Filtered Signal'):
    """
    Plot original and filtered signals for comparison.
    
    Parameters:
    -----------
    original : ndarray
        Original signal
    filtered : ndarray
        Filtered signal
    start_idx : int
        Starting index for visualization
    duration_samples : int, optional
        Number of samples to plot (if None, plot all)
    title : str
        Plot title
    """
    if duration_samples is None:
        duration_samples = len(original) - start_idx
    
    end_idx = start_idx + duration_samples
    
    plt.figure(figsize=(14, 6))
    plt.plot(original[start_idx:end_idx], alpha=0.7, label='Original', linewidth=1)
    plt.plot(filtered[start_idx:end_idx], alpha=0.9, label='Filtered', linewidth=2)
    plt.xlabel('Samples')
    plt.ylabel('Number of Cars')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_monday_start(dates, after_sample=1000):
    """
    Find the index of a Monday after a given sample number.
    
    Parameters:
    -----------
    dates : list
        List of datetime objects
    after_sample : int
        Start searching after this sample
        
    Returns:
    --------
    int
        Index of first Monday after after_sample
    """
    for i in range(after_sample, len(dates)):
        # Check if it's Monday (weekday() == 0) and hour is 00:00
        if dates[i].weekday() == 0 and dates[i].hour == 0:
            return i
    return after_sample


def interpret_frequency(freq, period_hours, period_days):
    """
    Interpret what a frequency component represents.
    
    Parameters:
    -----------
    freq : float
        Frequency in cycles/hour
    period_hours : float
        Period in hours
    period_days : float
        Period in days
        
    Returns:
    --------
    str
        Interpretation of the frequency
    """
    if np.isclose(period_days, 1, rtol=0.1):
        return "Daily cycle (day/night pattern)"
    elif np.isclose(period_days, 7, rtol=0.1):
        return "Weekly cycle (weekday/weekend pattern)"
    elif np.isclose(period_hours, 12, rtol=0.1):
        return "12-hour cycle (morning/evening peaks)"
    elif period_days < 1:
        return f"Sub-daily cycle ({period_hours:.1f} hours)"
    elif period_days > 7:
        return f"Long-term cycle ({period_days:.1f} days)"
    else:
        return f"Multi-day cycle ({period_days:.1f} days)"


# ============================================================================
# MAIN EXERCISES
# ============================================================================

def main():
    """Execute all lab exercises."""
    
    print("="*70)
    print("LAB 5 - FOURIER TRANSFORM ANALYSIS")
    print("="*70)
    
    # Load data
    signal, dates = load_traffic_data('Train.csv')
    print(f"Loaded {len(signal)} samples ({dates[0]} to {dates[-1]})")
    
    # Exercise 1a-c: Sampling information
    print("\n" + "-" * 70)
    print("EXERCISE 1a-c: Sampling Information")
    print("-" * 70)
    info = get_sampling_info(signal, sampling_period_hours=1)
    print(f"Sampling frequency: {info['Fs']} samples/hour")
    print(f"Total duration: {info['T']/24:.1f} days ({info['N']} samples)")
    print(f"Max frequency (Nyquist): {info['fmax']} cycles/hour")
    
    # Exercise 1d: Compute and plot FFT
    print("\n" + "-" * 70)
    print("EXERCISE 1d: FFT Computation")
    print("-" * 70)
    freqs, magnitude, X_full = compute_fft(signal, info['Fs'])
    print(f"Frequency resolution: {freqs[1]:.6f} cycles/hour")
    
    # Plot full signal
    plot_full_signal(signal, 'Traffic Signal - Full Dataset')
    
    # Plot FFT spectrum
    plot_fft_spectrum(freqs, magnitude, 'FFT Spectrum - Original Signal')
    
    # Exercise 1e: DC component
    print("\n" + "-" * 70)
    print("EXERCISE 1e: DC Component")
    print("-" * 70)
    has_dc, dc_value = has_dc_component(magnitude, threshold=1.0)
    
    if has_dc:
        signal_no_dc, dc_comp = remove_dc_component(signal)
        print(f"DC component detected: {dc_comp:.2f} (removed)")
        
        # Recompute FFT without DC
        freqs_no_dc, magnitude_no_dc, _ = compute_fft(signal_no_dc, info['Fs'])
        plot_fft_spectrum(freqs_no_dc, magnitude_no_dc, 
                         'FFT Spectrum - DC Removed')
        
        signal_analysis = signal_no_dc
        magnitude_analysis = magnitude_no_dc
    else:
        print("No significant DC component detected")
        signal_analysis = signal
        magnitude_analysis = magnitude
    
    # Exercise 1f: Top frequencies
    print("\n" + "-" * 70)
    print("EXERCISE 1f: Top 4 Frequency Components")
    print("-" * 70)
    top_freqs = find_top_frequencies(freqs, magnitude_analysis, n_top=4)
    
    for i, (freq, mag, period_h, period_d) in enumerate(top_freqs, 1):
        interpretation = interpret_frequency(freq, period_h, period_d)
        print(f"{i}. f={freq:.6f} Hz | Period={period_d:.1f}d | {interpretation}")
    
    # Plot FFT spectrum with highlighted peaks
    plot_fft_spectrum(freqs, magnitude_analysis, 
                     'FFT Spectrum - Top 4 Peaks Highlighted',
                     highlight_peaks=top_freqs)
    
    # Exercise 1g: Visualize one month starting from a Monday
    print("\n" + "-" * 70)
    print("EXERCISE 1g: One Month Visualization")
    print("-" * 70)
    monday_idx = find_monday_start(dates, after_sample=1000)
    one_month_samples = 24 * 30  # 30 days * 24 hours
    print(f"Plotting 30 days from {dates[monday_idx].strftime('%Y-%m-%d (%A)')}")
    
    plot_signal_segment(signal_analysis, monday_idx, one_month_samples, dates,
                       'Traffic Signal - One Month')
    
    # Exercise 1h: Method to determine start date
    print("\n" + "-" * 70)
    print("EXERCISE 1h: Determining Start Date")
    print("-" * 70)
    print("Method: Analyze weekly patterns in FFT, identify weekend low-traffic periods,")
    print("        and count backwards using the 7-day cycle.")
    print("Limitation: Cannot determine absolute date without external reference,")
    print("            only day of week. Affected by holidays and seasonal variations.")
    
    # Exercise 1i: Filter high frequencies
    print("\n" + "-" * 70)
    print("EXERCISE 1i: High Frequency Filtering")
    print("-" * 70)
    
    cutoff_freq = 1 / 12  # Remove periods shorter than 12 hours
    print(f"Cutoff: {cutoff_freq:.4f} cycles/hour (period < 12h)")
    print("Removes: hourly noise and outliers")
    print("Keeps: daily and weekly patterns")
    
    signal_filtered = filter_high_frequencies(signal_analysis, info['Fs'], cutoff_freq)
    
    # Plot comparison
    plot_comparison(signal_analysis, signal_filtered, 
                   start_idx=monday_idx, duration_samples=one_month_samples,
                   title='Original vs Filtered Signal')
    
    # Plot filtered signal FFT
    freqs_filt, magnitude_filt, _ = compute_fft(signal_filtered, info['Fs'])
    plot_fft_spectrum(freqs_filt, magnitude_filt, 
                     'FFT Spectrum - After Filtering')
    
    print("\n" + "="*70)
    print("LAB 5 COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
