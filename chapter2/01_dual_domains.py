import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. SINYAL 1D: DOMAIN WAKTU vs FREKUENSI
    fs_1d = 1000  # Sampling rate 1000 Hz
    t = np.linspace(0, 1.0, fs_1d) # Durasi 1 detik
    
    # Membuat sinyal gabungan: Nada 50 Hz (amplitudo 1) + Nada 120 Hz (amplitudo 0.5)
    f1, f2 = 50.0, 120.0
    signal_1d = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

    # Melakukan Fast Fourier Transform (FFT) 1D
    fft_1d = np.fft.fft(signal_1d)
    # Mengambil nilai magnitudo (absolut) dan menormalkannya
    mag_1d = np.abs(fft_1d) / (fs_1d / 2)
    # Membuat sumbu X untuk frekuensi (hanya ambil setengah bagian positif)
    freqs_1d = np.fft.fftfreq(fs_1d, 1/fs_1d)
    
    half_n = fs_1d // 2
    freqs_1d_pos = freqs_1d[:half_n]
    mag_1d_pos = mag_1d[:half_n]

    # 2. SINYAL 2D: DOMAIN RUANG vs FREKUENSI SPASIAL
    N = 256 # Resolusi gambar 256x256 piksel
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    
    # Membuat Sinusoidal Grating (garis-garis vertikal)
    # Frekuensi spasial = 10 siklus per gambar
    fx_2d = 10.0
    signal_2d = np.sin(2 * np.pi * fx_2d * X)

    # Melakukan FFT 2D
    fft_2d = np.fft.fft2(signal_2d)
    # Menggeser titik frekuensi nol (DC) ke tengah gambar
    fft_2d_shifted = np.fft.fftshift(fft_2d)
    mag_2d = np.abs(fft_2d_shifted)

    # 3. VISUALISASI HASIL
    fig = plt.figure(figsize=(14, 10))

    # --- Baris 1: Sinyal 1D ---
    # Domain Waktu
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t, signal_1d, 'b-')
    ax1.set_title('Domain Waktu 1D (Sinyal Asli)')
    ax1.set_xlabel('Waktu (detik)')
    ax1.set_ylabel('Amplitudo')
    ax1.set_xlim(0, 0.1)
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Domain Frekuensi
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(freqs_1d_pos, mag_1d_pos, 'r-')
    ax2.set_title('Domain Frekuensi 1D (Spektrum Magnitudo FFT)')
    ax2.set_xlabel('Frekuensi (Hz)')
    ax2.set_ylabel('Magnitudo')
    ax2.set_xlim(0, 200) # Hanya tampilkan sampai 200 Hz
    ax2.grid(True, linestyle=':', alpha=0.7)

    # --- Baris 2: Sinyal 2D ---
    # Domain Ruang (Citra)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(signal_2d, cmap='gray', extent=[0, 1, 0, 1])
    ax3.set_title('Domain Ruang 2D (Sinusoidal Grating)')
    ax3.set_xlabel('Sumbu X (Ruang)')
    ax3.set_ylabel('Sumbu Y (Ruang)')

    # Domain Frekuensi Spasial
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(mag_2d, cmap='gray', extent=[-N/2, N/2, -N/2, N/2])
    ax4.set_title('Domain Frekuensi 2D (Spektrum Magnitudo FFT2)')
    ax4.set_xlabel('Frekuensi X')
    ax4.set_ylabel('Frekuensi Y')
    
    # Menambahkan garis bantu silang di tengah FFT 2D
    ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")