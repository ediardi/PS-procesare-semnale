import numpy as np
import time
import matplotlib.pyplot as plt

# Laboratorul 4: Transformata Fourier - Partea II

def dft(x):
    """
    Calculeaza Transformata Fourier Discreta (DFT) a unui semnal.
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

def fft(x):
    """
    Calculeaza Transformata Fourier Rapida (FFT) a unui semnal.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        raise ValueError("Dimensiunea trebuie sa fie o putere a lui 2")
    elif N <= 32:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:int(N / 2)] * X_odd,
                               X_even + factor[int(N / 2):] * X_odd])

def ex1():
    """
    Exercitiul 1: Compararea timpilor de executie DFT, FFT si numpy.fft.
    """
    N_values = [128, 256, 512, 1024, 2048, 4096, 8192]
    dft_times = []
    fft_times = []
    numpy_fft_times = []

    for N in N_values:
        x = np.random.rand(N)

        # Timp executie DFT
        start_time = time.time()
        dft(x)
        dft_times.append(time.time() - start_time)

        # Timp executie FFT
        start_time = time.time()
        fft(x)
        fft_times.append(time.time() - start_time)

        # Timp executie numpy.fft
        start_time = time.time()
        np.fft.fft(x)
        numpy_fft_times.append(time.time() - start_time)

    plt.figure(figsize=(10, 6))
    plt.plot(N_values, dft_times, 'o-', label='DFT Implementat')
    plt.plot(N_values, fft_times, 'o-', label='FFT Implementat')
    plt.plot(N_values, numpy_fft_times, 'o-', label='numpy.fft')
    plt.xlabel('Dimensiune Vector (N)')
    plt.ylabel('Timp de Executie (s)')
    plt.title('Comparare Timpi de Executie DFT vs FFT vs numpy.fft')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('lab4/ex1_timpi_executie.pdf')
    plt.show()

def ex2_3():
    """
    Exercitiul 2 & 3: Fenomenul de aliere si esantionare corecta.
    """
    f = 50  # Frecventa semnalului original in Hz
    t = np.linspace(0, 0.1, 1000)
    signal = np.cos(2 * np.pi * f * t)

    # Exercitiul 2: Esantionare sub-Nyquist
    fs_sub = 80  # Frecventa de esantionare sub-Nyquist (fs < 2f)
    n_sub = np.arange(0, 0.1, 1 / fs_sub)
    samples_sub = np.cos(2 * np.pi * f * n_sub)

    # Semnale alias
    f_alias1 = f + fs_sub
    f_alias2 = f - fs_sub
    signal_alias1 = np.cos(2 * np.pi * f_alias1 * t)
    signal_alias2 = np.cos(2 * np.pi * f_alias2 * t)


    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label=f'Semnal Original ({f} Hz)')
    plt.plot(t, signal_alias1, '--', label=f'Semnal Alias 1 ({f_alias1} Hz)')
    plt.plot(t, signal_alias2, '--', label=f'Semnal Alias 2 ({f_alias2} Hz)')
    plt.stem(n_sub, samples_sub, 'r', markerfmt='ro', basefmt=' ', label=f'Esantioane (fs={fs_sub} Hz)')
    plt.title('Exercitiul 2: Fenomenul de Aliere (Esantionare sub-Nyquist)')
    plt.xlabel('Timp (s)')
    plt.ylabel('Amplitudine')
    plt.legend()
    plt.grid(True)
    plt.savefig('lab4/ex2_aliasing.pdf')


    # Exercitiul 3: Esantionare peste-Nyquist
    fs_nyquist = 2.5 * f # Frecventa de esantionare peste-Nyquist (fs > 2f)
    n_nyquist = np.arange(0, 0.1, 1 / fs_nyquist)
    samples_nyquist = np.cos(2 * np.pi * f * n_nyquist)
    samples_alias1_nyquist = np.cos(2 * np.pi * f_alias1 * n_nyquist)
    samples_alias2_nyquist = np.cos(2 * np.pi * f_alias2 * n_nyquist)


    plt.subplot(2, 1, 2)
    plt.plot(t, signal, label=f'Semnal Original ({f} Hz)')
    plt.stem(n_nyquist, samples_nyquist, 'g', markerfmt='go', basefmt=' ', label=f'Esantioane Semnal Original (fs={fs_nyquist} Hz)')
    plt.stem(n_nyquist, samples_alias1_nyquist, 'b', markerfmt='bo', basefmt=' ', linefmt='b--', label=f'Esantioane Semnal Alias 1 (fs={fs_nyquist} Hz)')
    plt.stem(n_nyquist, samples_alias2_nyquist, 'm', markerfmt='mo', basefmt=' ', linefmt='m--', label=f'Esantioane Semnal Alias 2 (fs={fs_nyquist} Hz)')
    plt.title('Exercitiul 3: Esantionare Peste-Nyquist')
    plt.xlabel('Timp (s)')
    plt.ylabel('Amplitudine')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lab4/ex3_no_aliasing.pdf')
    plt.show()

def ex6(file_path='lab2/semnal_cosinus.wav'):
    """
    Exercitiul 6: Generarea unei spectrograme dintr-un fisier audio.
    """
    try:
        from scipy.io import wavfile
        from scipy import signal
    except ImportError:
        print("Exercitiul 6 necesita scipy. Instaleaza cu: pip install scipy")
        return

    try:
        samplerate, data = wavfile.read(file_path)
        if data.ndim > 1:
            data = data.mean(axis=1) # Convert to mono

        # Parametri pentru spectrograma
        nperseg = int(len(data) * 0.01) # 1% din semnal
        noverlap = int(nperseg * 0.5) # 50% suprapunere

        frequencies, times, Sxx = signal.spectrogram(data, samplerate, nperseg=nperseg, noverlap=noverlap)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-9), shading='gouraud')
        plt.ylabel('Frecventa [Hz]')
        plt.xlabel('Timp [s]')
        plt.title('Exercitiul 6: Spectrograma')
        plt.colorbar(label='Intensitate [dB]')
        plt.savefig('lab4/ex6_spectrograma.pdf')
        plt.show()

    except FileNotFoundError:
        print(f"Fisierul audio '{file_path}' nu a fost gasit pentru exercitiul 6.")
    except Exception as e:
        print(f"A aparut o eroare la exercitiul 6: {e}")


def ex7():
    """
    Exercitiul 7: Calculul puterii zgomotului.
    """
    P_semnal_dB = 90
    SNR_dB = 80

    # SNR_dB = 10 * log10(P_semnal / P_zgomot)
    # SNR_dB = 10 * log10(P_semnal) - 10 * log10(P_zgomot)
    # P_semnal_dB = 10 * log10(P_semnal)
    # SNR_dB = P_semnal_dB - P_zgomot_dB
    P_zgomot_dB = P_semnal_dB - SNR_dB
    print(f"Exercitiul 7: Puterea zgomotului este {P_zgomot_dB} dB.")


if __name__ == '__main__':
    ex1()
    ex2_3()
    ex6()
    ex7()