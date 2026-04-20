import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import warnings

warnings.filterwarnings('ignore')

def main():
    # 1. PENGATURAN SINYAL
    fs = 1000      # Frekuensi sampling 1000 Hz
    N = 1000       # Jumlah sampel (1 detik).
    t = np.arange(N) / fs

    # Sinyal 1: Pas di bin
    f_on = 50.0 
    s_on = np.sin(2 * np.pi * f_on * t)

    # Sinyal 2: Tidak pas di bin
    f_off = 48.5
    s_off = np.sin(2 * np.pi * f_off * t)

    # 2. DEFINISI 4 JENIS WINDOW
    windows = {
        'Rectangular Window': np.ones(N),
        'Hann Window': get_window('hann', N),
        'Hamming Window': get_window('hamming', N),
        'Blackman Window': get_window('blackman', N)
    }

    # 3. VISUALISASI HASIL FFT
    plt.figure(figsize=(14, 8))
    plt.suptitle('Effect of Windowing on Spectral Leakage', fontsize=16, fontweight='bold')

    # Frekuensi sumbu X (hanya ambil setengah bagian positif)
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]

    for i, (name, win) in enumerate(windows.items(), 1):
        # Mengalikan sinyal dengan window masing-masing
        s_on_win = s_on * win
        s_off_win = s_off * win

        # Menghitung FFT dan melakukan normalisasi amplitudo berdasarkan area window
        # agar tinggi puncak gelombang tetap sebanding dan mudah dianalisis
        fft_on = np.abs(np.fft.fft(s_on_win))[:N//2] * 2 / np.sum(win)
        fft_off = np.abs(np.fft.fft(s_off_win))[:N//2] * 2 / np.sum(win)

        # Plotting
        plt.subplot(2, 2, i)
        
        # Plot garis sinyal (on bin dan off bin)
        plt.plot(freqs, fft_on, 'b-', linewidth=1.5, label='Signal on bin (50.0 Hz)')
        plt.plot(freqs, fft_off, 'r-', linewidth=1.5, alpha=0.8, label='Signal off bin (48.5 Hz)')
        
        plt.title(name)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        
        # Zoom di area 40 Hz hingga 60 Hz agar leakage terlihat jelas
        plt.xlim(40, 60)
        
        # Sumbu Y diatur logaritmik (dB-like shape visual) untuk memperjelas kebocoran di bawah
        plt.yscale('log')
        plt.ylim(1e-4, 1.5)
        
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")