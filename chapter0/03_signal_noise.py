import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Parameter Sinyal Dasar
    fs = 1000          # Frekuensi sampling (Hz)
    t = np.arange(0, 1.0, 1/fs)  # Waktu dari 0 sampai 1 detik
    f_signal = 5       # Frekuensi sinyal (5 Hz)
    
    # Sinyal bersih (Sinusoida murni)
    s_clean = np.sin(2 * np.pi * f_signal * t)

    # 2. Pembuatan Berbagai Jenis Noise
    std_dev = 0.5
    num_samples = len(t)
    
    # A. Gaussian Noise
    noise_gaussian = np.random.normal(loc=0.0, scale=std_dev, size=num_samples)
    
    # B. Uniform Noise
    range_uniform = np.sqrt(12) * std_dev / 2
    noise_uniform = np.random.uniform(low=-range_uniform, high=range_uniform, size=num_samples)
    
    # C. Laplace Noise
    b_laplace = std_dev / np.sqrt(2)
    noise_laplace = np.random.laplace(loc=0.0, scale=b_laplace, size=num_samples)

    # 3. Mencampur Sinyal dengan Noise
    s_noisy_gaussian = s_clean + noise_gaussian
    s_noisy_uniform = s_clean + noise_uniform
    s_noisy_laplace = s_clean + noise_laplace

    # 4. Visualisasi
    plt.figure(figsize=(15, 10))

    # --- Kolom Kiri: Sinyal di Domain Waktu ---
    plt.subplot(3, 2, 1)
    plt.plot(t, s_clean, 'k-', linewidth=2, label='Sinyal Bersih')
    plt.plot(t, s_noisy_gaussian, 'r-', alpha=0.6, label='Gaussian Noise')
    plt.title('Sinyal + Gaussian Noise')
    plt.ylabel('Amplitudo')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(3, 2, 3)
    plt.plot(t, s_clean, 'k-', linewidth=2)
    plt.plot(t, s_noisy_uniform, 'g-', alpha=0.6, label='Uniform Noise')
    plt.title('Sinyal + Uniform Noise')
    plt.ylabel('Amplitudo')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(3, 2, 5)
    plt.plot(t, s_clean, 'k-', linewidth=2)
    plt.plot(t, s_noisy_laplace, 'b-', alpha=0.6, label='Laplace Noise')
    plt.title('Sinyal + Laplace Noise')
    plt.xlabel('Waktu (detik)')
    plt.ylabel('Amplitudo')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    # --- Kolom Kanan: Distribusi (Histogram) Noise ---
    bins = 30
    
    plt.subplot(3, 2, 2)
    plt.hist(noise_gaussian, bins=bins, color='red', alpha=0.7, density=True)
    plt.title('Distribusi Gaussian (Lonceng)')
    plt.ylabel('Kepadatan Probabilitas')
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(3, 2, 4)
    plt.hist(noise_uniform, bins=bins, color='green', alpha=0.7, density=True)
    plt.title('Distribusi Uniform (Rata/Kotak)')
    plt.ylabel('Kepadatan Probabilitas')
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(3, 2, 6)
    plt.hist(noise_laplace, bins=bins, color='blue', alpha=0.7, density=True)
    plt.title('Distribusi Laplace (Tajam di Tengah)')
    plt.xlabel('Nilai Noise')
    plt.ylabel('Kepadatan Probabilitas')
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")