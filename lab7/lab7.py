from scipy import datasets, ndimage
import numpy as np
import matplotlib.pyplot as plt

X = datasets.face(gray=True)
plt.imshow(X, cmap=plt.cm.gray)
plt.show()


Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))

plt.imshow(freq_db)
plt.colorbar()
plt.show()

rotate_angle = 45
X45 = ndimage.rotate(X, rotate_angle)
plt.imshow(X45, cmap=plt.cm.gray)
plt.show()

Y45 = np.fft.fft2(X45)
plt.imshow(20*np.log10(abs(Y45)))
plt.colorbar()
plt.show()


freq_x = np.fft.fftfreq(X.shape[1])
freq_y = np.fft.fftfreq(X.shape[0])

plt.stem(freq_x, freq_db[:][0])
plt.show()


freq_cutoff = 120

Y_cutoff = Y.copy()
Y_cutoff[freq_db > freq_cutoff] = 0
X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                # in practice use irfft2
plt.imshow(X_cutoff, cmap=plt.cm.gray)
plt.show()

pixel_noise = 200

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.show()
plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title('Noisy')
plt.show()


# Task 1
N = 100
n1 = np.arange(N)
n2 = np.arange(N)
n1_grid, n2_grid = np.meshgrid(n1, n2, indexing='ij')

# 1. x = sin(2*pi*n1 + 3*pi*n2)
# Since n1, n2 are integers, sin(k*pi) = 0.
x1 = np.sin(2*np.pi*n1_grid + 3*np.pi*n2_grid)

# 2. x = sin(4*pi*n1) + cos(6*pi*n2)
# sin(4*pi*n1) = 0, cos(6*pi*n2) = 1.
x2 = np.sin(4*np.pi*n1_grid) + np.cos(6*np.pi*n2_grid)

# 3. Y peaks at (0,5) and (0, N-5)
Y3 = np.zeros((N, N), dtype=complex)
Y3[0, 5] = 1
Y3[0, N-5] = 1
x3 = np.fft.ifft2(Y3).real

# 4. Y peaks at (5,0) and (N-5, 0)
Y4 = np.zeros((N, N), dtype=complex)
Y4[5, 0] = 1
Y4[N-5, 0] = 1
x4 = np.fft.ifft2(Y4).real

# 5. Y peaks at (5,5) and (N-5, N-5)
Y5 = np.zeros((N, N), dtype=complex)
Y5[5, 5] = 1
Y5[N-5, N-5] = 1
x5 = np.fft.ifft2(Y5).real

# Plotting
def plot_pair(x, title_prefix):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(x, cmap='gray')
    axs[0].set_title(f'{title_prefix} - Image')
    
    Y = np.fft.fft2(x)
    Y_mag = np.abs(Y)
    # Log scale, handle 0
    Y_log = 20 * np.log10(Y_mag + 1e-9)
    
    axs[1].imshow(Y_log, cmap='viridis')
    axs[1].set_title(f'{title_prefix} - Spectrum (dB)')
    plt.show()

plot_pair(x1, '1. sin(2pi*n1 + 3pi*n2)')
plot_pair(x2, '2. sin(4pi*n1) + cos(6pi*n2)')
plot_pair(x3, '3. Y peaks at (0,5)')
plot_pair(x4, '4. Y peaks at (5,0)')
plot_pair(x5, '5. Y peaks at (5,5)')

# Task 2: Compress image by attenuating high frequencies until SNR threshold
try:
    X
except NameError:
    # Fallback if X is not defined (e.g. if previous cells weren't run)
    # We need to import datasets again just in case
    from scipy import datasets
    X = datasets.face(gray=True)

def calculate_snr(original, noisy):
    # SNR = 10 * log10( ||original||^2 / ||original - noisy||^2 )
    signal_power = np.sum(original**2)
    noise_power = np.sum((original - noisy)**2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

target_snr = 20 # dB
print(f"Target SNR: {target_snr} dB")

Y = np.fft.fft2(X)
Y_shifted = np.fft.fftshift(Y) # Shift DC to center
rows, cols = X.shape
center_row, center_col = rows // 2, cols // 2

# We will use a circular mask radius as the cutoff parameter
# Start with full radius (no compression) and decrease
max_radius = int(np.sqrt(center_row**2 + center_col**2))
best_radius = max_radius
final_snr = float('inf')
X_compressed = X.copy()

# Iterate to find the smallest radius that satisfies SNR >= target_snr
# Or rather, the problem says "attenuate ... UP TO a self-imposed SNR threshold".
# This implies we want to compress AS MUCH AS POSSIBLE while keeping SNR >= threshold.
# So we want the SMALLEST radius that gives SNR >= target.

for radius in range(max_radius, 0, -5):
    # Create mask
    y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
    mask = x*x + y*y <= radius*radius
    
    Y_masked_shifted = Y_shifted * mask
    Y_masked = np.fft.ifftshift(Y_masked_shifted)
    X_recon = np.fft.ifft2(Y_masked).real
    
    snr = calculate_snr(X, X_recon)
    
    if snr < target_snr:
        # SNR dropped below target, so the previous radius was the limit
        # But since we want "up to", maybe this is the point where we stop.
        # Let's keep the last valid one or just stop here.
        print(f"SNR dropped to {snr:.2f} dB at radius {radius}. Stopping.")
        break
    
    best_radius = radius
    final_snr = snr
    X_compressed = X_recon

print(f"Compression complete. Radius: {best_radius}, Final SNR: {final_snr:.2f} dB")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap='gray')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(X_compressed, cmap='gray')
plt.title(f'Compressed (SNR={final_snr:.2f} dB)')
plt.show()

# Task 3: Remove noise and present SNR before and after
pixel_noise = 200
# Re-generate noise to be consistent with the task description
noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise

snr_noisy = calculate_snr(X, X_noisy)
print(f"SNR of Noisy Image: {snr_noisy:.2f} dB")

# Denoise using Low Pass Filter (Frequency Domain)
# We search for a radius that maximizes SNR(X, X_denoised)
Y_noisy = np.fft.fft2(X_noisy)
Y_noisy_shifted = np.fft.fftshift(Y_noisy)

best_denoised_snr = -float('inf')
best_denoised_radius = 0
X_denoised_best = X_noisy.copy()

# Search range: from small radius (heavy blur) to full image
# We expect an optimum somewhere in between
for radius in range(10, max_radius, 5):
    y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
    mask = x*x + y*y <= radius*radius
    
    Y_filtered_shifted = Y_noisy_shifted * mask
    Y_filtered = np.fft.ifftshift(Y_filtered_shifted)
    X_denoised = np.fft.ifft2(Y_filtered).real
    
    current_snr = calculate_snr(X, X_denoised)
    
    if current_snr > best_denoised_snr:
        best_denoised_snr = current_snr
        best_denoised_radius = radius
        X_denoised_best = X_denoised

print(f"Best Denoising Radius: {best_denoised_radius}")
print(f"SNR after Denoising: {best_denoised_snr:.2f} dB")

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(X, cmap='gray')
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(X_noisy, cmap='gray')
plt.title(f'Noisy (SNR={snr_noisy:.2f} dB)')
plt.subplot(1, 3, 3)
plt.imshow(X_denoised_best, cmap='gray')
plt.title(f'Denoised (SNR={best_denoised_snr:.2f} dB)')
plt.show()