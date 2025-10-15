"""
Procesarea Semnalelor - Laboratorul 2
Generarea și manipularea semnalelor audio
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import scipy.signal
import sounddevice

print("Laboratorul 2 - Procesarea Semnalelor")
print("=" * 50)

# Ghid Python - Funcții pentru lucrul cu audio

def save_audio_signal(signal, filename, sample_rate=44100):
    """
    Salvează un semnal generat în format audio
    
    Args:
        signal: numpy array cu semnalul
        filename: numele fișierului (ex: 'nume.wav')
        sample_rate: frecvența de eșantionare (implicit 44100 Hz)
    """
    rate = int(sample_rate)
    scipy.io.wavfile.write(filename, rate, signal)
    print(f"Semnalul a fost salvat în {filename}")

def load_audio_signal(filename):
    """
    Încarcă un semnal salvat anterior pentru procesare
    
    Args:
        filename: numele fișierului audio
    
    Returns:
        rate: frecvența de eșantionare
        signal: numpy array cu semnalul
    """
    rate, x = scipy.io.wavfile.read(filename)
    print(f"Semnalul a fost încărcat din {filename}")
    print(f"Frecvența de eșantionare: {rate} Hz")
    return rate, x

def play_audio_signal(signal, fs=44100):
    """
    Redă audio un semnal salvat într-un numpy.array
    
    Args:
        signal: numpy array cu semnalul
        fs: frecvența de eșantionare (implicit 44100 Hz)
    """
    sounddevice.play(signal, fs)
    print(f"Se redă semnalul cu frecvența de eșantionare {fs} Hz")

# Exercițiul 1: Generarea de semnale sinusoidale

print("\n" + "=" * 50)
print("Exercițiul 1: Semnale sinusoidale identice")
print("=" * 50)

# Parametrii pentru semnale
duration = 2.0  # durata în secunde
fs = 44100      # frecvența de eșantionare
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Parametrii aleși pentru semnale
amplitude = 1.0      # amplitudinea
frequency = 440.0    # frecvența în Hz (nota La)
phase_sin = 0.0      # faza pentru sinus

print(f"Parametrii semnalelor:")
print(f"- Amplitudine: {amplitude}")
print(f"- Frecvența: {frequency} Hz")
print(f"- Durata: {duration} s")
print(f"- Frecvența de eșantionare: {fs} Hz")

# Generarea semnalului sinus
signal_sin = amplitude * np.sin(2 * np.pi * frequency * t + phase_sin)

# Pentru ca semnalul cosinus să fie identic cu semnalul sinus,
# folosim relația: sin(x) = cos(x - π/2)
# Deci: sin(2πft + φ_sin) = cos(2πft + φ_sin - π/2)
# Aceasta înseamnă că: φ_cos = φ_sin - π/2
phase_cos = phase_sin - np.pi/2

# Generarea semnalului cosinus
signal_cos = amplitude * np.cos(2 * np.pi * frequency * t + phase_cos)

print(f"\nFaza sinus: {phase_sin:.4f} rad ({np.degrees(phase_sin):.1f}°)")
print(f"Faza cosinus: {phase_cos:.4f} rad ({np.degrees(phase_cos):.1f}°)")

# Verificarea că semnalele sunt identice
difference = np.abs(signal_sin - signal_cos)
max_difference = np.max(difference)
print(f"\nDiferența maximă între semnale: {max_difference:.2e}")

if max_difference < 1e-10:
    print("✓ Semnalele sunt identice!")
else:
    print("✗ Semnalele nu sunt identice.")

# Afișarea grafică în două subplot-uri diferite
plt.figure(figsize=(12, 8))

# Afișăm doar primele 0.01 secunde pentru claritate
t_display = t[:int(0.01 * fs)]
sin_display = signal_sin[:int(0.01 * fs)]
cos_display = signal_cos[:int(0.01 * fs)]

# Subplot pentru semnalul sinus
plt.subplot(2, 1, 1)
plt.plot(t_display * 1000, sin_display, 'b-', linewidth=2, label=f'Sinus: {amplitude}·sin(2π·{frequency}·t + {phase_sin:.3f})')
plt.title('Semnalul Sinus')
plt.xlabel('Timp (ms)')
plt.ylabel('Amplitudine')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 10)

# Subplot pentru semnalul cosinus
plt.subplot(2, 1, 2)
plt.plot(t_display * 1000, cos_display, 'r-', linewidth=2, label=f'Cosinus: {amplitude}·cos(2π·{frequency}·t + {phase_cos:.3f})')
plt.title('Semnalul Cosinus')
plt.xlabel('Timp (ms)')
plt.ylabel('Amplitudine')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 10)

plt.tight_layout()
plt.show()

# Compararea directă în același grafic
plt.figure(figsize=(12, 6))
plt.plot(t_display * 1000, sin_display, 'b-', linewidth=2, label='Sinus', alpha=0.7)
plt.plot(t_display * 1000, cos_display, 'r--', linewidth=2, label='Cosinus', alpha=0.7)
plt.title('Compararea semnalelor Sinus și Cosinus')
plt.xlabel('Timp (ms)')
plt.ylabel('Amplitudine')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 10)
plt.show()

# Salvarea semnalelor audio (opțional)
save_choice = input("\nDoriți să salvați semnalele audio? (y/n): ")
if save_choice.lower() == 'y':
    # Normalizarea semnalelor pentru salvare audio
    signal_sin_norm = np.int16(signal_sin * 32767)
    signal_cos_norm = np.int16(signal_cos * 32767)
    
    save_audio_signal(signal_sin_norm, 'lab2/semnal_sinus.wav', fs)
    save_audio_signal(signal_cos_norm, 'lab2/semnal_cosinus.wav', fs)
    
    # Redarea audio (opțional)
    play_choice = input("Doriți să redați semnalul sinus? (y/n): ")
    if play_choice.lower() == 'y':
        print("Se redă semnalul sinus pentru 2 secunde...")
        play_audio_signal(signal_sin, fs)
        sounddevice.wait()  # Așteaptă să se termine redarea

print("\n" + "=" * 50)
print("Exercițiul 2: Semnale cu faze diferite și zgomot")
print("=" * 50)

# Parametrii pentru semnale
duration_ex2 = 1.0  # durata în secunde pentru exercițiul 2
fs_ex2 = 1000       # frecvența de eșantionare redusă pentru vizualizare mai clară
t_ex2 = np.linspace(0, duration_ex2, int(fs_ex2 * duration_ex2), endpoint=False)

# Parametrii pentru semnalul sinusoidal
amplitude_ex2 = 1.0      # amplitudine unitară
frequency_ex2 = 10.0     # frecvența aleasă (10 Hz)

# 4 valori diferite pentru fază
phases = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°
phase_names = ['0°', '45°', '90°', '135°']

print(f"Parametrii semnalelor:")
print(f"- Amplitudine: {amplitude_ex2}")
print(f"- Frecvența: {frequency_ex2} Hz")
print(f"- Durata: {duration_ex2} s")
print(f"- Frecvența de eșantionare: {fs_ex2} Hz")
print(f"- Faze testate: {phase_names}")

# Generarea semnalelor cu faze diferite
signals_phases = []
for i, phase in enumerate(phases):
    signal = amplitude_ex2 * np.sin(2 * np.pi * frequency_ex2 * t_ex2 + phase)
    signals_phases.append(signal)

# Afișarea tuturor semnalelor pe același grafic
plt.figure(figsize=(12, 8))
colors = ['blue', 'red', 'green', 'orange']

for i, (signal, phase, phase_name, color) in enumerate(zip(signals_phases, phases, phase_names, colors)):
    plt.plot(t_ex2, signal, color=color, linewidth=2, 
             label=f'Faza {phase_name} ({phase:.3f} rad)', alpha=0.8)

plt.title('Semnale sinusoidale cu faze diferite')
plt.xlabel('Timp (s)')
plt.ylabel('Amplitudine')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 0.5)  # Afișăm primele 0.5 secunde pentru claritate
plt.show()

# Partea cu zgomotul - folosim primul semnal (faza 0°)
print(f"\nAdăugarea zgomotului la semnalul cu faza 0°:")
x = signals_phases[0]  # semnalul original

# Generarea zgomotului Gaussian
np.random.seed(42)  # pentru reproducibilitate
z = np.random.normal(0, 1, len(x))  # zgomot Gaussian standard

# Valorile SNR dorite
snr_values = [0.1, 1, 10, 100]

# Calcularea valorilor γ pentru fiecare SNR
print(f"\nCalculul parametrului γ pentru fiecare SNR:")

# Calcularea normelor
norm_x = np.linalg.norm(x)
norm_z = np.linalg.norm(z)

print(f"||x||₂ = {norm_x:.4f}")
print(f"||z||₂ = {norm_z:.4f}")

gamma_values = []
noisy_signals = []

for snr in snr_values:
    # Din formula SNR = ||x||²₂ / (γ² ||z||²₂)
    # Rezultă: γ = ||x||₂ / (√SNR * ||z||₂)
    gamma = norm_x / (np.sqrt(snr) * norm_z)
    gamma_values.append(gamma)
    
    # Generarea semnalului cu zgomot
    noisy_signal = x + gamma * z
    noisy_signals.append(noisy_signal)
    
    # Verificarea SNR-ului calculat
    calculated_snr = (norm_x**2) / ((gamma**2) * (norm_z**2))
    
    print(f"SNR = {snr:5.1f} → γ = {gamma:.6f} → SNR verificat = {calculated_snr:.4f}")

# Afișarea semnalelor cu zgomot
plt.figure(figsize=(15, 10))

# Semnalul original
plt.subplot(2, 3, 1)
plt.plot(t_ex2, x, 'b-', linewidth=2, label='Semnal original')
plt.title('Semnalul original (fără zgomot)')
plt.xlabel('Timp (s)')
plt.ylabel('Amplitudine')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 0.5)

# Semnalele cu zgomot
for i, (snr, gamma, noisy_signal) in enumerate(zip(snr_values, gamma_values, noisy_signals)):
    plt.subplot(2, 3, i + 2)
    plt.plot(t_ex2, noisy_signal, 'r-', linewidth=1, alpha=0.8, label=f'Semnal + zgomot')
    plt.plot(t_ex2, x, 'b-', linewidth=2, alpha=0.6, label='Semnal original')
    plt.title(f'SNR = {snr} (γ = {gamma:.4f})')
    plt.xlabel('Timp (s)')
    plt.ylabel('Amplitudine')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 0.5)

plt.tight_layout()
plt.show()

# Compararea tuturor semnalelor cu zgomot pe același grafic
plt.figure(figsize=(12, 8))

# Semnalul original
plt.plot(t_ex2, x, 'k-', linewidth=3, label='Semnal original', alpha=0.8)

# Semnalele cu zgomot
colors_noise = ['red', 'orange', 'green', 'blue']
for i, (snr, noisy_signal, color) in enumerate(zip(snr_values, noisy_signals, colors_noise)):
    plt.plot(t_ex2, noisy_signal, color=color, linewidth=1.5, 
             label=f'SNR = {snr}', alpha=0.7)

plt.title('Compararea semnalelor cu zgomot pentru diferite valori SNR')
plt.xlabel('Timp (s)')
plt.ylabel('Amplitudine')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 0.5)
plt.show()

# Salvarea semnalelor cu zgomot (opțional)
save_noise_choice = input("\nDoriți să salvați semnalele cu zgomot? (y/n): ")
if save_noise_choice.lower() == 'y':
    for i, (snr, noisy_signal) in enumerate(zip(snr_values, noisy_signals)):
        # Normalizarea pentru salvare audio
        signal_norm = np.int16(noisy_signal * 32767 / np.max(np.abs(noisy_signal)))
        filename = f'lab2/semnal_zgomot_SNR_{snr}.wav'
        save_audio_signal(signal_norm, filename, fs_ex2)

print("\n" + "=" * 50)
print("Laboratorul 2 - Exercițiile 1 și 2 completate!")
print("=" * 50)