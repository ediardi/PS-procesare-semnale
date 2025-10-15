"""
Procesarea Semnalelor - Laboratorul 1
Introducere în semnale continue și discrete
"""

import matplotlib.pyplot as plt
import numpy as np

print("Laboratorul 1 - Procesarea Semnalelor")
print("=" * 50)

# Exercițiul 1: Semnale continue x(t), y(t), z(t)

# (a) Simularea axei reale de timp
# Interval de timp [0:0.0005:0.03] - pas de 0.0005 secunde
t = np.arange(0, 0.03, 0.0005)  # Vector de timp
print(f"\n1(a) Axa de timp:")
print(f"Intervalul de timp: {t[0]:.4f} - {t[-1]:.4f} secunde")
print(f"Pasul de timp: {t[1]-t[0]:.4f} secunde")
print(f"Numărul de puncte: {len(t)}")

# (b) Construirea semnalelor continue
print(f"\n1(b) Semnale continue:")
x_t = np.cos(520 * np.pi * t + np.pi/3)  # x(t) = cos(520πt + π/3)
y_t = np.cos(280 * np.pi * t - np.pi/3)  # y(t) = cos(280πt - π/3)
z_t = np.cos(120 * np.pi * t + np.pi/3)  # z(t) = cos(120πt + π/3)

print("x(t) = cos(520πt + π/3)")
print("y(t) = cos(280πt - π/3)")
print("z(t) = cos(120πt + π/3)")

# Calculul frecvențelor în Hz
f_x = 520 / 2  # Frecvența semnalului x în Hz
f_y = 280 / 2  # Frecvența semnalului y în Hz
f_z = 120 / 2  # Frecvența semnalului z în Hz

print(f"\nFrecvențele semnalelor:")
print(f"f_x = {f_x} Hz")
print(f"f_y = {f_y} Hz")
print(f"f_z = {f_z} Hz")

# Afișarea graficelor pentru semnalele continue
fig1, axs1 = plt.subplots(3, 1, figsize=(12, 10))
fig1.suptitle('Semnale Continue', fontsize=16, fontweight='bold')

axs1[0].plot(t, x_t, 'b-', linewidth=1.5, label='x(t)')
axs1[0].set_title('x(t) = cos(520πt + π/3), f = 260 Hz')
axs1[0].set_xlabel('Timp (s)')
axs1[0].set_ylabel('Amplitudine')
axs1[0].grid(True, alpha=0.3)
axs1[0].legend()

axs1[1].plot(t, y_t, 'r-', linewidth=1.5, label='y(t)')
axs1[1].set_title('y(t) = cos(280πt - π/3), f = 140 Hz')
axs1[1].set_xlabel('Timp (s)')
axs1[1].set_ylabel('Amplitudine')
axs1[1].grid(True, alpha=0.3)
axs1[1].legend()

axs1[2].plot(t, z_t, 'g-', linewidth=1.5, label='z(t)')
axs1[2].set_title('z(t) = cos(120πt + π/3), f = 60 Hz')
axs1[2].set_xlabel('Timp (s)')
axs1[2].set_ylabel('Amplitudine')
axs1[2].grid(True, alpha=0.3)
axs1[2].legend()

# Setarea limitelor pentru toate subplot-urile
for ax in axs1.flat:
    ax.set_xlim([0, 0.03])
    ax.set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.savefig('lab1/semnale_continue.eps', format='eps', bbox_inches='tight')
plt.savefig('lab1/semnale_continue.pdf', format='pdf', bbox_inches='tight')

# (c) Eșantionarea semnalelor cu frecvența de 200 Hz
print(f"\n1(c) Eșantionarea semnalelor:")
fs = 200  # Frecvența de eșantionare (Hz)
Ts = 1/fs  # Perioada de eșantionare
print(f"Frecvența de eșantionare: {fs} Hz")
print(f"Perioada de eșantionare: {Ts:.4f} secunde")

# Generarea indicilor pentru eșantionare
n_samples = int(0.03 * fs) + 1  # Numărul de eșantioane pentru 0.03 secunde
n = np.arange(n_samples)  # Indicii eșantioanelor [0, 1, 2, ..., n_samples-1]
t_sampled = n * Ts  # Momentele de timp eșantionate

print(f"Numărul de eșantioane: {n_samples}")
print(f"Intervalul de eșantionare: {t_sampled[0]:.4f} - {t_sampled[-1]:.4f} secunde")

# Semnalele eșantionate: x[n] = x(nT), y[n] = y(nT), z[n] = z(nT)
x_n = np.cos(520 * np.pi * t_sampled + np.pi/3)  # x[n]
y_n = np.cos(280 * np.pi * t_sampled - np.pi/3)  # y[n]
z_n = np.cos(120 * np.pi * t_sampled + np.pi/3)  # z[n]

# Afișarea graficelor pentru semnalele eșantionate
fig2, axs2 = plt.subplots(3, 1, figsize=(12, 10))
fig2.suptitle('Semnale Eșantionate (fs = 200 Hz)', fontsize=16, fontweight='bold')

axs2[0].stem(t_sampled, x_n, basefmt=' ', linefmt='b-', markerfmt='bo')
axs2[0].set_title('x[n] - Eșantionat la 200 Hz')
axs2[0].set_xlabel('Timp (s)')
axs2[0].set_ylabel('Amplitudine')
axs2[0].grid(True, alpha=0.3)

axs2[1].stem(t_sampled, y_n, basefmt=' ', linefmt='r-', markerfmt='ro')
axs2[1].set_title('y[n] - Eșantionat la 200 Hz')
axs2[1].set_xlabel('Timp (s)')
axs2[1].set_ylabel('Amplitudine')
axs2[1].grid(True, alpha=0.3)

axs2[2].stem(t_sampled, z_n, basefmt=' ', linefmt='g-', markerfmt='go')
axs2[2].set_title('z[n] - Eșantionat la 200 Hz')
axs2[2].set_xlabel('Timp (s)')
axs2[2].set_ylabel('Amplitudine')
axs2[2].grid(True, alpha=0.3)

# Setarea limitelor pentru toate subplot-urile
for ax in axs2.flat:
    ax.set_xlim([0, 0.03])
    ax.set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.savefig('lab1/semnale_esantionate.eps', format='eps', bbox_inches='tight')
plt.savefig('lab1/semnale_esantionate.pdf', format='pdf', bbox_inches='tight')

# Afișarea graficelor
plt.show()

# Informații suplimentare despre eșantionare
print(f"\nInformații despre eșantionare:")
print(f"Teorema de eșantionare (Nyquist): fs >= 2 * f_max")
print(f"Frecvența maximă în semnale: {max(f_x, f_y, f_z)} Hz")
print(f"Frecvența minimă Nyquist necesară: {2 * max(f_x, f_y, f_z)} Hz")
print(f"Frecvența de eșantionare folosită: {fs} Hz")

if fs >= 2 * max(f_x, f_y, f_z):
    print("✓ Condiția Nyquist este respectată - eșantionarea este corectă")
else:
    print("✗ Condiția Nyquist NU este respectată - va apărea aliasing!")

print(f"\nFișierele salvate:")
print("- semnale_continue.eps")
print("- semnale_continue.pdf")
print("- semnale_esantionate.eps")
print("- semnale_esantionate.pdf")

# ========================================================================
# Exercițiul 2: Generarea și afișarea diferitelor tipuri de semnale
# ========================================================================

print("\n" + "="*60)
print("Exercițiul 2 - Generarea diferitelor tipuri de semnale")
print("="*60)

# (a) Semnal sinusoidal de 400 Hz cu 1600 de eșantioane
print("\n2(a) Semnal sinusoidal 400 Hz, 1600 eșantioane:")
f_sin = 400  # Frecvența în Hz
N_samples = 1600  # Numărul de eșantioane
fs_sin = f_sin * 4  # Frecvența de eșantionare (4x frecvența semnalului pentru vizualizare bună)
T_sin = N_samples / fs_sin  # Durata totală
t_sin = np.linspace(0, T_sin, N_samples, endpoint=False)  # Vector de timp
sin_signal = np.sin(2 * np.pi * f_sin * t_sin)  # Semnalul sinusoidal

print(f"Frecvența: {f_sin} Hz")
print(f"Numărul de eșantioane: {N_samples}")
print(f"Frecvența de eșantionare: {fs_sin} Hz")
print(f"Durata: {T_sin:.3f} secunde")

# (b) Semnal sinusoidal de 800 Hz care durează 3 secunde
print("\n2(b) Semnal sinusoidal 800 Hz, 3 secunde:")
f_sin2 = 800  # Frecvența în Hz
T_duration = 3.0  # Durata în secunde
fs_sin2 = f_sin2 * 4  # Frecvența de eșantionare
N_samples2 = int(fs_sin2 * T_duration)  # Numărul de eșantioane
t_sin2 = np.linspace(0, T_duration, N_samples2, endpoint=False)  # Vector de timp
sin_signal2 = np.sin(2 * np.pi * f_sin2 * t_sin2)  # Semnalul sinusoidal

print(f"Frecvența: {f_sin2} Hz")
print(f"Durata: {T_duration} secunde")
print(f"Frecvența de eșantionare: {fs_sin2} Hz")
print(f"Numărul de eșantioane: {N_samples2}")

# (c) Semnal sawtooth de 240 Hz
print("\n2(c) Semnal sawtooth 240 Hz:")
f_saw = 240  # Frecvența în Hz
fs_saw = f_saw * 10  # Frecvența de eșantionare
T_saw = 3.0  # Durata în secunde
t_saw = np.linspace(0, T_saw, int(fs_saw * T_saw), endpoint=False)
# Semnal sawtooth folosind numpy.mod
sawtooth_signal = 2 * (f_saw * t_saw - np.floor(f_saw * t_saw + 0.5))

print(f"Frecvența: {f_saw} Hz")
print(f"Durata: {T_saw} secunde")
print(f"Frecvența de eșantionare: {fs_saw} Hz")

# (d) Semnal square de 300 Hz
print("\n2(d) Semnal square 300 Hz:")
f_square = 300  # Frecvența în Hz
fs_square = f_square * 10  # Frecvența de eșantionare
T_square = 2.0  # Durata în secunde
t_square = np.linspace(0, T_square, int(fs_square * T_square), endpoint=False)
# Semnal square folosind numpy.sign
square_signal = np.sign(np.sin(2 * np.pi * f_square * t_square))

print(f"Frecvența: {f_square} Hz")
print(f"Durata: {T_square} secunde")
print(f"Frecvența de eșantionare: {fs_square} Hz")

# (e) Semnal 2D aleator 128x128
print("\n2(e) Semnal 2D aleator 128x128:")
size_2d = 128
random_2d = np.random.rand(size_2d, size_2d)
print(f"Dimensiuni: {random_2d.shape}")
print(f"Valoare minimă: {random_2d.min():.3f}")
print(f"Valoare maximă: {random_2d.max():.3f}")
print(f"Valoare medie: {random_2d.mean():.3f}")

# (f) Semnal 2D personalizat 128x128 - Model geometric
print("\n2(f) Semnal 2D personalizat 128x128 - Model geometric:")
# Creez un pattern geometric interesant
x_2d = np.linspace(-1, 1, size_2d)
y_2d = np.linspace(-1, 1, size_2d)
X, Y = np.meshgrid(x_2d, y_2d)

# Pattern cu cercuri concentrice și unde radiale
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
custom_2d = np.sin(8 * np.pi * R) * np.cos(6 * Theta) * np.exp(-2 * R)

print(f"Dimensiuni: {custom_2d.shape}")
print(f"Valoare minimă: {custom_2d.min():.3f}")
print(f"Valoare maximă: {custom_2d.max():.3f}")
print("Pattern: Cercuri concentrice cu unde radiale")

# Afișarea graficelor pentru toate semnalele
print(f"\nInformații despre primele 3 repetări:")
print("="*50)

# Figura pentru semnalele 1D
fig3, axs3 = plt.subplots(2, 2, figsize=(15, 10))
fig3.suptitle('Semnale 1D - Exercițiul 2 (Primele 3 Repetări)', fontsize=16, fontweight='bold')

# (a) Semnal sinusoidal 400 Hz - primele 3 repetări
T_period_sin = 1/f_sin  # Perioada semnalului
t_3_periods_sin = 3 * T_period_sin  # Timpul pentru 3 perioade
mask_sin = t_sin <= t_3_periods_sin  # Masca pentru primele 3 perioade
print(f"Perioada semnalului {f_sin} Hz: {T_period_sin:.4f} s")
print(f"Timpul pentru 3 perioade: {t_3_periods_sin:.4f} s")
print(f"Numărul de puncte afișate: {np.sum(mask_sin)}")

axs3[0, 0].plot(t_sin[mask_sin], sin_signal[mask_sin], 'b-', linewidth=1.5)
axs3[0, 0].set_title(f'(a) Sinusoidal {f_sin} Hz - Primele 3 repetări')
axs3[0, 0].set_xlabel('Timp (s)')
axs3[0, 0].set_ylabel('Amplitudine')
axs3[0, 0].grid(True, alpha=0.3)

# (b) Semnal sinusoidal 800 Hz - primele 3 repetări
T_period_sin2 = 1/f_sin2  # Perioada semnalului
t_3_periods_sin2 = 3 * T_period_sin2  # Timpul pentru 3 perioade
mask_sin2 = t_sin2 <= t_3_periods_sin2  # Masca pentru primele 3 perioade
print(f"Perioada semnalului {f_sin2} Hz: {T_period_sin2:.6f} s")
print(f"Timpul pentru 3 perioade: {t_3_periods_sin2:.6f} s")
print(f"Numărul de puncte afișate: {np.sum(mask_sin2)}")

axs3[0, 1].plot(t_sin2[mask_sin2], sin_signal2[mask_sin2], 'r-', linewidth=1.5)
axs3[0, 1].set_title(f'(b) Sinusoidal {f_sin2} Hz - Primele 3 repetări')
axs3[0, 1].set_xlabel('Timp (s)')
axs3[0, 1].set_ylabel('Amplitudine')
axs3[0, 1].grid(True, alpha=0.3)

# (c) Semnal sawtooth - primele 3 repetări
T_period_saw = 1/f_saw  # Perioada semnalului
t_3_periods_saw = 3 * T_period_saw  # Timpul pentru 3 perioade
mask_saw = t_saw <= t_3_periods_saw  # Masca pentru primele 3 perioade
print(f"Perioada semnalului sawtooth {f_saw} Hz: {T_period_saw:.6f} s")
print(f"Timpul pentru 3 perioade: {t_3_periods_saw:.6f} s")
print(f"Numărul de puncte afișate: {np.sum(mask_saw)}")

axs3[1, 0].plot(t_saw[mask_saw], sawtooth_signal[mask_saw], 'g-', linewidth=1.5)
axs3[1, 0].set_title(f'(c) Sawtooth {f_saw} Hz - Primele 3 repetări')
axs3[1, 0].set_xlabel('Timp (s)')
axs3[1, 0].set_ylabel('Amplitudine')
axs3[1, 0].grid(True, alpha=0.3)

# (d) Semnal square - primele 3 repetări
T_period_square = 1/f_square  # Perioada semnalului
t_3_periods_square = 3 * T_period_square  # Timpul pentru 3 perioade
mask_square = t_square <= t_3_periods_square  # Masca pentru primele 3 perioade
print(f"Perioada semnalului square {f_square} Hz: {T_period_square:.6f} s")
print(f"Timpul pentru 3 perioade: {t_3_periods_square:.6f} s")
print(f"Numărul de puncte afișate: {np.sum(mask_square)}")

axs3[1, 1].plot(t_square[mask_square], square_signal[mask_square], 'm-', linewidth=1.5)
axs3[1, 1].set_title(f'(d) Square {f_square} Hz - Primele 3 repetări')
axs3[1, 1].set_xlabel('Timp (s)')
axs3[1, 1].set_ylabel('Amplitudine')
axs3[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab1/semnale_1d_ex2.eps', format='eps', bbox_inches='tight')
plt.savefig('lab1/semnale_1d_ex2.pdf', format='pdf', bbox_inches='tight')

# Figura pentru semnalele 2D
fig4, axs4 = plt.subplots(1, 2, figsize=(12, 5))
fig4.suptitle('Semnale 2D - Exercițiul 2', fontsize=16, fontweight='bold')

# (e) Semnal 2D aleator
im1 = axs4[0].imshow(random_2d, cmap='viridis', aspect='equal')
axs4[0].set_title('(e) Semnal 2D aleator 128x128')
axs4[0].set_xlabel('Coloană')
axs4[0].set_ylabel('Linie')
plt.colorbar(im1, ax=axs4[0], fraction=0.046, pad=0.04)

# (f) Semnal 2D personalizat
im2 = axs4[1].imshow(custom_2d, cmap='RdBu_r', aspect='equal')
axs4[1].set_title('(f) Pattern geometric 128x128')
axs4[1].set_xlabel('Coloană')
axs4[1].set_ylabel('Linie')
plt.colorbar(im2, ax=axs4[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('lab1/semnale_2d_ex2.eps', format='eps', bbox_inches='tight')
plt.savefig('lab1/semnale_2d_ex2.pdf', format='pdf', bbox_inches='tight')

# Afișarea tuturor graficelor
plt.show()

print(f"\nFișierele noi salvate pentru exercițiul 2:")
print("- semnale_1d_ex2.eps")
print("- semnale_1d_ex2.pdf")
print("- semnale_2d_ex2.eps")
print("- semnale_2d_ex2.pdf")
print("\nToate graficele au fost generate și salvate în format vectorial!")

# ========================================================================
# Exercițiul 3: Calcule de digitizare și stocare
# ========================================================================

print("\n" + "="*60)
print("Exercițiul 3 - Calcule de digitizare și stocare")
print("="*60)

# Date inițiale
fs_digitizare = 2000  # Frecvența de eșantionare în Hz
bits_per_sample = 4   # Numărul de biți per eșantion
ore_achizitie = 1     # Durata achiziției în ore

print(f"\nDate inițiale:")
print(f"Frecvența de eșantionare: {fs_digitizare} Hz")
print(f"Biți per eșantion: {bits_per_sample} biți")
print(f"Durata achiziției: {ore_achizitie} oră(e)")

# (a) Intervalul de timp între două eșantioane
print(f"\n3(a) Intervalul de timp între două eșantioane:")
Ts_digitizare = 1 / fs_digitizare  # Perioada de eșantionare în secunde
Ts_ms = Ts_digitizare * 1000       # Perioada în milisecunde
Ts_us = Ts_digitizare * 1_000_000  # Perioada în microsecunde

print(f"Ts = 1/fs = 1/{fs_digitizare} = {Ts_digitizare} secunde")
print(f"Ts = {Ts_ms} milisecunde")
print(f"Ts = {Ts_us} microsecunde")

# (b) Calculul spațiului de stocare pentru 1 oră
print(f"\n3(b) Spațiul de stocare pentru {ore_achizitie} oră de achiziție:")

# Calculul pas cu pas
secunde_per_ora = 3600  # secunde într-o oră
secunde_totale = ore_achizitie * secunde_per_ora
esantioane_totale = secunde_totale * fs_digitizare
biti_totali = esantioane_totale * bits_per_sample
bytes_totali = biti_totali / 8  # 8 biți = 1 byte

print(f"\nCalculul pas cu pas:")
print(f"1. Secunde în {ore_achizitie} oră: {secunde_totale} secunde")
print(f"2. Eșantioane totale: {secunde_totale} s × {fs_digitizare} Hz = {esantioane_totale:,} eșantioane")
print(f"3. Biți totali: {esantioane_totale:,} eșantioane × {bits_per_sample} biți = {biti_totali:,} biți")
print(f"4. Bytes totali: {biti_totali:,} biți ÷ 8 = {bytes_totali:,.0f} bytes")

# Conversii în unități mai mari
KB = bytes_totali / 1024
MB = KB / 1024
GB = MB / 1024

print(f"\nConversii în unități mai mari:")
print(f"• {bytes_totali:,.0f} bytes")
print(f"• {KB:,.2f} KB (kilobytes)")
print(f"• {MB:,.2f} MB (megabytes)")
print(f"• {GB:,.4f} GB (gigabytes)")

# Verificarea calculului cu o formulă directă
bytes_directa = (fs_digitizare * bits_per_sample * secunde_totale) / 8
print(f"\nVerificare cu formula directă:")
print(f"Bytes = (fs × biți_per_eșantion × timp_secunde) ÷ 8")
print(f"Bytes = ({fs_digitizare} × {bits_per_sample} × {secunde_totale}) ÷ 8 = {bytes_directa:,.0f} bytes")

# Informații suplimentare utile
print(f"\nInformații suplimentare:")
print(f"• Rate de date: {fs_digitizare * bits_per_sample:,} biți/secundă = {(fs_digitizare * bits_per_sample)/8:,.0f} bytes/secundă")
print(f"• Rezoluție: 2^{bits_per_sample} = {2**bits_per_sample} nivele posibile")
print(f"• Intervalul valorilor: 0 până la {2**bits_per_sample - 1}")

# Comparație cu alte formate
print(f"\nComparație cu alte rezoluții:")
for biti in [8, 16, 24, 32]:
    bytes_alt = (fs_digitizare * biti * secunde_totale) / 8
    MB_alt = bytes_alt / (1024 * 1024)
    print(f"• {biti} biți/eșantion: {MB_alt:,.2f} MB pentru 1 oră")

print(f"\n" + "="*60)
print("Rezumatul exercițiului 3:")
print(f"(a) Intervalul între eșantioane: {Ts_digitizare} s = {Ts_ms} ms")
print(f"(b) Spațiu pentru 1 oră: {bytes_totali:,.0f} bytes = {MB:,.2f} MB")
print("="*60)