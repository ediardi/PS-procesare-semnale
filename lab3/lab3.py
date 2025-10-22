import os
import numpy as np
import matplotlib.pyplot as plt


def fourier_matrix(N: int) -> np.ndarray:
    """
    Construct the (unnormalized) NxN discrete Fourier transform matrix F
    with entries F[k, n] = exp(-2πj k n / N).

    Returns complex ndarray of shape (N, N).
    """
    n = np.arange(N)
    k = n.reshape((N, 1))
    return np.exp(-2j * np.pi * k * n / N)


def verify_unitarity(F: np.ndarray) -> dict:
    """
    Verify unitarity properties for the Fourier matrix.

    For the unnormalized DFT matrix F, we have F^H F = N * I.
    For the unitary-normalized matrix Fu = F / sqrt(N), Fu^H Fu = I.

    Returns a dict with boolean checks and norms for diagnostics.
    """
    N = F.shape[0]
    identity = np.eye(N, dtype=complex)

    FhF = F.conj().T @ F
    check_unnorm = np.allclose(FhF, N * identity, atol=1e-10)
    err_unnorm = np.linalg.norm(FhF - N * identity)

    Fu = F / np.sqrt(N)
    FuhFu = Fu.conj().T @ Fu
    check_unitary = np.allclose(FuhFu, identity, atol=1e-10)
    err_unitary = np.linalg.norm(FuhFu - identity)

    return {
        "unnormalized_allclose_NI": check_unnorm,
        "unnormalized_error_norm": float(err_unnorm),
        "unitary_allclose_I": check_unitary,
        "unitary_error_norm": float(err_unitary),
    }


def plot_rows_real_imag(F: np.ndarray, out_path: str) -> None:
    """
    For each row of F, plot real and imaginary parts on separate subplots
    and save the figure to out_path.
    """
    N = F.shape[0]
    x = np.arange(N)

    fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(10, 2 * N), constrained_layout=True)
    if N == 1:
        axes = np.array([[axes]])  # normalize shape for N=1 edge case

    for r in range(N):
        # Real part
        ax_real = axes[r, 0]
        markerline, stemlines, baseline = ax_real.stem(x, F[r].real, linefmt="C0-", markerfmt="C0o")
        ax_real.set_title(f"Row {r} — Real")
        ax_real.set_xlabel("n")
        ax_real.set_ylabel("Re{F[r, n]}")
        ax_real.grid(True, alpha=0.3)

        # Imag part
        ax_imag = axes[r, 1]
        markerline, stemlines, baseline = ax_imag.stem(x, F[r].imag, linefmt="C1-", markerfmt="C1o")
        ax_imag.set_title(f"Row {r} — Imag")
        ax_imag.set_xlabel("n")
        ax_imag.grid(True, alpha=0.3)

    fig.suptitle("DFT Fourier Matrix (N=8) — Rows: Real vs Imag", fontsize=14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    # -----------------
    # Exercise 1: DFT F
    # -----------------
    N = 8
    F = fourier_matrix(N)

    # Verify unitarity properties
    results = verify_unitarity(F)
    print("Unitarity checks for DFT matrix (N=8):")
    print(f"  F^H F == N*I (allclose): {results['unnormalized_allclose_NI']}  | error norm = {results['unnormalized_error_norm']:.3e}")
    print(f"  (F/√N)^H (F/√N) == I (allclose): {results['unitary_allclose_I']}  | error norm = {results['unitary_error_norm']:.3e}")

    # Prepare output path alongside this script
    this_dir = os.path.dirname(os.path.abspath(__file__))
    out_path_F = os.path.join(this_dir, "fourier_matrix_rows.png")

    # Plot real/imag for each row and save
    plot_rows_real_imag(F, out_path_F)
    print(f"Saved row plots to: {out_path_F}")

    # -----------------
    # Exercise 2: Figures 1 and 2
    # -----------------
    fs = 1000  # Hz (sampling rate)
    duration = 1.0  # seconds
    f0 = 5  # Hz (choose a different frequency vs guide)

    t = np.arange(0, duration, 1.0 / fs)
    n = np.arange(t.size)  # sample index
    x = np.sin(2 * np.pi * f0 * t)

    # Figure 1: left x(t); right wrapping with ω=1 Hz
    omega_wrap_fig1 = 15  # Hz
    y = x * np.exp(-2j * np.pi * omega_wrap_fig1 * n / fs)

    fig1, (ax_sig, ax_wrap) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    # Left: time-domain signal
    ax_sig.plot(t, x, 'C0-')
    ax_sig.set_title(f"Semnal sinusoidal: f0 = {f0:g} Hz, fs = {fs} Hz")
    ax_sig.set_xlabel("Timp (s)")
    ax_sig.set_ylabel("Amplitudine")
    ax_sig.grid(True, alpha=0.3)

    # Right: wrapping on the unit circle with color by |y|
    # Draw unit circle
    phi = np.linspace(0, 2 * np.pi, 512)
    ax_wrap.plot(np.cos(phi), np.sin(phi), color='0.8', linewidth=1.0, label="Cercul unitate")
    sc1 = ax_wrap.scatter(y.real, y.imag, c=np.abs(y), cmap="viridis", s=8, alpha=0.9)
    ax_wrap.set_title(f"Înfășurare pe cerc (ω = {omega_wrap_fig1:g} Hz)")
    ax_wrap.set_xlabel("Real")
    ax_wrap.set_ylabel("Imaginar")
    ax_wrap.set_aspect('equal', adjustable='box')
    ax_wrap.grid(True, alpha=0.3)
    cbar1 = fig1.colorbar(sc1, ax=ax_wrap)
    cbar1.set_label("Distanța față de origine = |y[n]|")

    out_fig1 = os.path.join(this_dir, "fig1_signal_wrapping.png")
    fig1.savefig(out_fig1, dpi=150)
    plt.close(fig1)
    print(f"Saved Figure 1 to: {out_fig1}")

    # Figure 2: z[ω] = x[n] e^{-2π j ω n / fs} for different ω, include ω=f0
    omegas = [2.0, 4.0, f0, 11.0]  # Hz (one equals the signal frequency)
    fig2, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    axes = axes.ravel()
    for idx, omega in enumerate(omegas):
        z = x * np.exp(-2j * np.pi * omega * n / fs)
        ax = axes[idx]
        # Draw unit circle
        ax.plot(np.cos(phi), np.sin(phi), color='0.85', linewidth=1.0)
        sc = ax.scatter(z.real, z.imag, c=np.abs(z), cmap="viridis", s=8, alpha=0.9)
        ax.set_title(f"z[ω] (ω = {omega:g} Hz)")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginar")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

    # One shared colorbar for all subplots
    cbar2 = fig2.colorbar(sc, ax=axes.tolist(), shrink=0.9)
    cbar2.set_label("Distanța față de origine = |z[n]| = |x[n]|")

    out_fig2 = os.path.join(this_dir, "fig2_wrap_frequencies.png")
    fig2.savefig(out_fig2, dpi=150)
    plt.close(fig2)
    print(f"Saved Figure 2 to: {out_fig2}")

    # -----------------
    # Exercise 3: Multi-tone signal and manual Fourier magnitude (like Fig. 3)
    # -----------------
    # Compose a signal with 3 frequency components (pick your own set)
    components = [
        (1.0, 6.0),   # amplitude, frequency (Hz)
        (2.0, 18.0),
        (0.5, 55.0),
    ]
    x3 = np.zeros_like(t)
    for a, f in components:
        x3 += a * np.cos(2 * np.pi * f * t)

    # Compute transform magnitudes |X(f)| using the relation analogous to (1),
    # sampled on a frequency grid in Hz: X(f) = sum_n x[n] e^{-j 2π f n / fs}
    N3 = x3.size
    n_idx = np.arange(N3)
    highest_f = max(f for _, f in components)
    fmax = min(int(max(100, 2 * highest_f)), int(fs // 2))
    freqs_hz = np.arange(0, fmax + 1)  # 0..fmax Hz, step 1 Hz
    W = np.exp(-2j * np.pi * (freqs_hz.reshape(-1, 1) * n_idx.reshape(1, -1)) / fs)
    Xf = W @ x3
    magX = np.abs(Xf)

    fig3, (ax3_sig, ax3_mag) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    # Left: time-domain multi-tone signal
    ax3_sig.plot(t, x3, 'C0-')
    ax3_sig.set_title("Semnal compus (3 componente)")
    ax3_sig.set_xlabel("Timp (s)")
    ax3_sig.set_ylabel("Amplitudine")
    ax3_sig.grid(True, alpha=0.3)

    # Right: |X(f)| vs frequency (Hz)
    markerline, stemlines, baseline = ax3_mag.stem(freqs_hz, magX, linefmt='C2-', markerfmt='C2o', basefmt='k-')
    ax3_mag.set_xlim(0, fmax)
    ax3_mag.set_title("|X(f)| (transformata Fourier calculată manual)")
    ax3_mag.set_xlabel("Frecvență (Hz)")
    ax3_mag.set_ylabel("|X(f)|")
    ax3_mag.grid(True, alpha=0.3)

    out_fig3 = os.path.join(this_dir, "fig3_multi_tone_spectrum.png")
    fig3.savefig(out_fig3, dpi=150)
    plt.close(fig3)
    print(f"Saved Figure 3 (multi-tone spectrum) to: {out_fig3}")
    
    
    


if __name__ == "__main__":
    main()