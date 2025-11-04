import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import scipy.signal
import sounddevice

def save_wav(signal, filename, fs=44100):
    scipy.io.wavfile.write(filename, fs, np.int16(signal * 32767))

def play_audio(signal, fs=44100):
    sounddevice.play(signal, fs)
    sounddevice.wait()

# Ex 1: Sinus vs Cosinus identic
fs, duration = 44100, 2.0
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
freq, amp = 440.0, 1.0

sig_sin = amp * np.sin(2 * np.pi * freq * t)
sig_cos = amp * np.cos(2 * np.pi * freq * t - np.pi/2)

print(f"Diferență max: {np.max(np.abs(sig_sin - sig_cos)):.2e}")

# Ex 2: Faze și zgomot
fs2, dur2 = 1000, 1.0
t2 = np.linspace(0, dur2, int(fs2 * dur2), endpoint=False)
phases = [0, np.pi/4, np.pi/2, 3*np.pi/4]

fig, ax = plt.subplots(figsize=(12, 6))
for phase in phases:
    sig = np.sin(2 * np.pi * 10 * t2 + phase)
    ax.plot(t2, sig, label=f'φ={phase:.2f}', alpha=0.7)
ax.set_xlim([0, 0.5])
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# Zgomot SNR
x = np.sin(2 * np.pi * 10 * t2)
z = np.random.normal(0, 1, len(x))
snr_vals = [0.1, 1, 10, 100]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.ravel()
for i, snr in enumerate(snr_vals):
    gamma = np.linalg.norm(x) / (np.sqrt(snr) * np.linalg.norm(z))
    noisy = x + gamma * z
    axs[i].plot(t2[:500], noisy[:500])
    axs[i].set_title(f'SNR = {snr}')
    axs[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Ex 4: Adunare semnale
fs4, dur4 = 8000, 1.0
t4 = np.linspace(0, dur4, int(fs4 * dur4), endpoint=False)
sig1 = np.sin(2 * np.pi * 5 * t4)
sig2 = 0.8 * scipy.signal.sawtooth(2 * np.pi * 7 * t4)
sig_sum = sig1 + sig2

fig, axs = plt.subplots(3, 1, figsize=(12, 8))
for i, (sig, label) in enumerate([(sig1, 'Sinus 5Hz'), (sig2, 'Sawtooth 7Hz'), (sig_sum, 'Suma')]):
    axs[i].plot(t4[:int(0.5*fs4)], sig[:int(0.5*fs4)])
    axs[i].set_title(label)
    axs[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Ex 5: Concatenare
freq1, freq2 = 261.63, 392.00
fs5, dur5 = 44100, 1.5
t5 = np.linspace(0, dur5, int(fs5 * dur5), endpoint=False)
s1 = 0.5 * np.sin(2 * np.pi * freq1 * t5)
s2 = 0.5 * np.sin(2 * np.pi * freq2 * t5)
s_concat = np.concatenate([s1, s2])

fig, ax = plt.subplots(figsize=(12, 4))
t_concat = np.arange(len(s_concat)) / fs5
ax.plot(t_concat, s_concat)
ax.axvline(x=dur5, color='r', linestyle='--', label='Tranziție')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# Ex 6: Frecvențe speciale
fs6, dur6 = 100, 2.0
t6 = np.linspace(0, dur6, int(fs6 * dur6), endpoint=False)

fig, axs = plt.subplots(3, 1, figsize=(12, 8))
freqs = [fs6/2, fs6/4, 0]
for i, f in enumerate(freqs):
    sig = np.sin(2 * np.pi * f * t6)
    axs[i].plot(t6[:20], sig[:20], 'o-')
    axs[i].set_title(f'f = {f:.1f} Hz')
    axs[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Ex 7: Decimare
fs7, dur7 = 1000, 1.0
t7 = np.linspace(0, dur7, int(fs7 * dur7), endpoint=False)
sig7 = np.sin(2 * np.pi * 5 * t7)

decimated_0 = sig7[::4]
decimated_1 = sig7[1::4]
t_dec = t7[::4]

fig, axs = plt.subplots(3, 1, figsize=(12, 8))
axs[0].plot(t7[:500], sig7[:500], 'b.', ms=2, label='Original')
axs[1].plot(t_dec[:125], decimated_0[:125], 'ro-', label='Start=0')
axs[2].plot(t7[1::4][:125], decimated_1[:125], 'go-', label='Start=1')
for ax in axs:
    ax.grid(True, alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.show()

# Ex 8: Aproximări sin(α)
alpha = np.linspace(-np.pi/2, np.pi/2, 1000)
sin_exact = np.sin(alpha)
sin_linear = alpha
sin_pade = (alpha - 7*alpha**3/60) / (1 + alpha**2/20)

fig, axs = plt.subplots(2, 1, figsize=(12, 8))
axs[0].plot(alpha, sin_exact, 'k-', linewidth=2, label='sin(α)')
axs[0].plot(alpha, sin_linear, 'b--', label='α')
axs[0].plot(alpha, sin_pade, 'r-.', label='Padé')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

axs[1].semilogy(alpha, np.abs(sin_exact - sin_linear), 'b-', label='Eroare liniară')
axs[1].semilogy(alpha, np.abs(sin_exact - sin_pade), 'r-', label='Eroare Padé')
axs[1].legend()
axs[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Lab 2 completat!")
