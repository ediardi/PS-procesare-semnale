import os
import numpy as np
import matplotlib.pyplot as plt

def fourier_matrix(N):
    n = np.arange(N)
    k = n.reshape((N, 1))
    return np.exp(-2j * np.pi * k * n / N)

def verify_unitarity(F):
    N = F.shape[0]
    identity = np.eye(N, dtype=complex)
    
    FhF = F.conj().T @ F
    err_unnorm = np.linalg.norm(FhF - N * identity)
    
    Fu = F / np.sqrt(N)
    err_unitary = np.linalg.norm(Fu.conj().T @ Fu - identity)
    
    return {
        "unnorm_error": float(err_unnorm),
        "unitary_error": float(err_unitary),
    }

def plot_rows(F, out_path):
    N = F.shape[0]
    x = np.arange(N)
    
    fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(10, 2*N))
    if N == 1:
        axes = np.array([[axes]])
    
    for r in range(N):
        axes[r, 0].stem(x, F[r].real, linefmt="C0-", markerfmt="C0o")
        axes[r, 0].set_title(f"Row {r} — Real")
        axes[r, 0].grid(True, alpha=0.3)
        
        axes[r, 1].stem(x, F[r].imag, linefmt="C1-", markerfmt="C1o")
        axes[r, 1].set_title(f"Row {r} — Imag")
        axes[r, 1].grid(True, alpha=0.3)
    
    fig.suptitle("DFT Fourier Matrix (N=8)", fontsize=14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# Ex 1: Matricea Fourier
N = 8
F = fourier_matrix(N)
results = verify_unitarity(F)
print(f"Eroare F^H F: {results['unnorm_error']:.3e}")
print(f"Eroare unitar: {results['unitary_error']:.3e}")

this_dir = os.path.dirname(os.path.abspath(__file__))
plot_rows(F, os.path.join(this_dir, "fourier_matrix.png"))

# Ex 2: Înfășurare semnal
fs, duration, f0 = 1000, 1.0, 5
t = np.arange(0, duration, 1.0/fs)
x = np.sin(2 * np.pi * f0 * t + np.pi/4)
n = np.arange(t.size)

omega_wrap = 15
y = x * np.exp(-2j * np.pi * omega_wrap * n / fs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(t, x)
ax1.set_title(f"Semnal: f0={f0}Hz, fs={fs}Hz")
ax1.grid(True, alpha=0.3)

phi = np.linspace(0, 2*np.pi, 512)
ax2.plot(np.cos(phi), np.sin(phi), '0.8')
ax2.scatter(y.real, y.imag, c=np.abs(y), cmap="viridis", s=8, alpha=0.9)
ax2.set_title(f"Înfășurare (ω={omega_wrap}Hz)")
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(this_dir, "signal_wrap.png"), dpi=150)
plt.show()

# Ex 3: Spectru multi-ton
components = [(1.0, 6.0), (2.0, 18.0), (0.5, 55.0)]
x3 = np.zeros_like(t)
for a, f in components:
    x3 += a * np.cos(2 * np.pi * f * t)

N3 = x3.size
n_idx = np.arange(N3)
fmax = 100
freqs_hz = np.arange(0, fmax + 1)
W = np.exp(-2j * np.pi * (freqs_hz.reshape(-1, 1) * n_idx.reshape(1, -1)) / fs)
Xf = W @ x3
magX = np.abs(Xf)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(t, x3)
ax1.set_title("Semnal compus (3 componente)")
ax1.grid(True, alpha=0.3)

ax2.stem(freqs_hz, magX, linefmt='C2-', markerfmt='C2o')
ax2.set_xlim(0, fmax)
ax2.set_title("|X(f)| - Transformata Fourier")
ax2.set_xlabel("Frecvență (Hz)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(this_dir, "spectrum.png"), dpi=150)
plt.show()

print("Lab 3 completat!")
