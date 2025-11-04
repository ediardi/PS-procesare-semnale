import matplotlib.pyplot as plt
import numpy as np

def plot_signals(t, signals, labels, title, filename):
    fig, axs = plt.subplots(len(signals), 1, figsize=(12, 8))
    if len(signals) == 1:
        axs = [axs]
    
    for ax, sig, label in zip(axs, signals, labels):
        ax.plot(t, sig, linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel('Timp (s)')
        ax.set_ylabel('Amplitudine')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, max(t)])
        ax.set_ylim([-1.2, 1.2])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'lab1/{filename}.pdf', bbox_inches='tight')
    plt.show()

# Ex 1: Semnale continue
t = np.arange(0, 0.03, 0.0005)
x_t = np.cos(520 * np.pi * t + np.pi/3)
y_t = np.cos(280 * np.pi * t - np.pi/3)
z_t = np.cos(120 * np.pi * t + np.pi/3)

plot_signals(t, [x_t, y_t, z_t], 
             ['x(t) - 260 Hz', 'y(t) - 140 Hz', 'z(t) - 60 Hz'],
             'Semnale Continue', 'semnale_continue')

# Eșantionare
fs = 200
t_s = np.arange(int(0.03 * fs) + 1) / fs
x_n = np.cos(520 * np.pi * t_s + np.pi/3)
y_n = np.cos(280 * np.pi * t_s - np.pi/3)
z_n = np.cos(120 * np.pi * t_s + np.pi/3)

fig, axs = plt.subplots(3, 1, figsize=(12, 8))
for i, (sig, label) in enumerate([(x_n, 'x[n]'), (y_n, 'y[n]'), (z_n, 'z[n]')]):
    axs[i].stem(t_s, sig, basefmt=' ')
    axs[i].set_title(f'{label} - Eșantionat la {fs} Hz')
    axs[i].set_xlabel('Timp (s)')
    axs[i].set_ylabel('Amplitudine')
    axs[i].grid(True, alpha=0.3)
    axs[i].set_xlim([0, 0.03])
    axs[i].set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.savefig('lab1/semnale_esantionate.pdf', bbox_inches='tight')
plt.show()

# Ex 2: Tipuri de semnale
fs2 = 4000
t2 = np.linspace(0, 3, int(fs2 * 3), endpoint=False)

signals_2 = {
    'sin_400': (np.sin(2 * np.pi * 400 * t2[:fs2]), 'Sinus 400 Hz'),
    'sin_800': (np.sin(2 * np.pi * 800 * t2[:int(3*fs2)]), 'Sinus 800 Hz'),
    'saw_240': (2 * (240 * t2 - np.floor(240 * t2 + 0.5)), 'Sawtooth 240 Hz'),
    'square_300': (np.sign(np.sin(2 * np.pi * 300 * t2[:int(2*fs2)])), 'Square 300 Hz')
}

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.ravel()
for i, (sig, label) in enumerate(signals_2.values()):
    n_show = min(int(0.03 * len(sig)), len(sig))
    axs[i].plot(np.linspace(0, len(sig)/fs2, len(sig))[:n_show], sig[:n_show])
    axs[i].set_title(label)
    axs[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab1/semnale_1d.pdf', bbox_inches='tight')
plt.show()

# Ex 2: Imagini 2D
random_2d = np.random.rand(128, 128)
x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
R = np.sqrt(x_grid**2 + y_grid**2)
custom_2d = np.sin(8 * np.pi * R) * np.exp(-2 * R)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(random_2d, cmap='viridis')
axs[0].set_title('Aleator 128x128')
axs[1].imshow(custom_2d, cmap='RdBu_r')
axs[1].set_title('Pattern 128x128')
plt.tight_layout()
plt.savefig('lab1/semnale_2d.pdf', bbox_inches='tight')
plt.show()

# Ex 3: Calcule digitizare
fs_dig = 2000
bits = 4
hours = 1
Ts = 1 / fs_dig
bytes_total = (fs_dig * bits * hours * 3600) / 8
print("\nEx 3: Digitizare")
print(f"Ts = {Ts*1000:.3f} ms")
print(f"Stocare 1h: {bytes_total/1024/1024:.2f} MB")
