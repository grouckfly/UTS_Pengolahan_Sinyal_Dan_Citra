import numpy as np
import matplotlib.pyplot as plt

def main():
    # Parameter Dasar
    fs = 1000
    t = np.linspace(-0.5, 0.5, fs)
    
    # Tiga variasi standar deviasi (lebar pulsa di domain waktu)
    sigmas = [0.005, 0.02, 0.1]
    labels = ['Sempit (Narrow Time)', 'Sedang (Medium)', 'Lebar (Wide Time)']
    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(14, 10))
    plt.suptitle('Visualisasi Prinsip Ketidakpastian Heisenberg', fontsize=16, fontweight='bold')

    for i, sigma in enumerate(sigmas):
        # 1. Membuat Pulsa Gaussian di Domain Waktu
        gauss_time = np.exp(-t**2 / (2 * sigma**2))
        
        # 2. Menghitung FFT (Domain Frekuensi)
        gauss_freq = np.fft.fftshift(np.fft.fft(gauss_time))
        freqs = np.fft.fftshift(np.fft.fftfreq(fs, 1/fs))
        
        # Normalisasi magnitudo agar puncaknya 1 (untuk perbandingan lebar)
        mag_freq = np.abs(gauss_freq) / np.max(np.abs(gauss_freq))

        # --- Plot Domain Waktu ---
        plt.subplot(2, 3, i + 1)
        plt.plot(t, gauss_time, color=colors[i], linewidth=2)
        plt.title(f'Waktu: {labels[i]}\n$\\sigma = {sigma}$')
        plt.xlabel('Waktu (detik)')
        plt.ylabel('Amplitudo')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xlim(-0.2, 0.2)

        # --- Plot Domain Frekuensi ---
        plt.subplot(2, 3, i + 4)
        plt.plot(freqs, mag_freq, color=colors[i], linewidth=2)
        plt.title(f'Frekuensi: Respon Spektral')
        plt.xlabel('Frekuensi (Hz)')
        plt.ylabel('Magnitudo')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xlim(-200, 200)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")