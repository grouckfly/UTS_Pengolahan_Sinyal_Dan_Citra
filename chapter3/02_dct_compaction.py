import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy.signal import chirp
import warnings

warnings.filterwarnings('ignore')

def main():
    # 1. PEMBUATAN 4 KARAKTERISTIK SINYAL
    N = 512  # Jumlah sampel
    t = np.linspace(0, 1, N)

    # A. Sinyal Sinus Mulus
    sig_sine = np.sin(2 * np.pi * 3 * t)

    # B. Sinyal Step Function
    sig_step = np.ones(N)
    sig_step[:N//2] = -1

    # C. Sinyal Chirp
    sig_chirp = chirp(t, f0=1, f1=50, t1=1, method='linear')

    # D. Sinyal White Noise
    np.random.seed(42)
    sig_noise = np.random.randn(N)

    signals = {
        'Sinus Mulus': sig_sine,
        'Step Function (Tepi Tajam)': sig_step,
        'Chirp (Frek. Naik)': sig_chirp,
        'White Noise (Acak)': sig_noise
    }

    # 2. VISUALISASI HASIL
    plt.figure(figsize=(15, 7))
    plt.suptitle('Analisis Kompaksi Energi DCT', fontsize=16, fontweight='bold')

    colors = ['blue', 'green', 'orange', 'red']

    # --- Plot Kiri: Bentuk Sinyal Waktu ---
    plt.subplot(1, 2, 1)
    for i, (name, sig) in enumerate(signals.items()):
        offset = 4 - i * 2.5 
        plt.plot(t, sig + offset, label=name, color=colors[i], linewidth=1.5)
        
    plt.title('Bentuk Sinyal (Domain Waktu)')
    plt.xlabel('Waktu')
    plt.yticks([])
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)

    # --- Plot Kanan: Kurva Akumulasi Energi ---
    plt.subplot(1, 2, 2)
    for i, (name, sig) in enumerate(signals.items()):
        # 1. Menghitung koefisien DCT
        dct_coeffs = dct(sig, norm='ortho')
        
        # 2. Menghitung energi tiap koefisien (dikuadratkan)
        energy = dct_coeffs**2
        
        # 3. Menghitung persentase kumulatif energinya
        cumulative_energy = np.cumsum(energy) / np.sum(energy) * 100
        
        plt.plot(range(N), cumulative_energy, label=name, color=colors[i], linewidth=2.5)

    plt.title('Kurva Kompaksi Energi DCT\n(Seberapa cepat energinya terkumpul?)')
    plt.xlabel('Jumlah Koefisien DCT Pertama yang Diambil')
    plt.ylabel('Persentase Energi Terakumulasi (%)')
    
    # Zoom ke 100 koefisien pertama (dari total 512) agar kurvanya terlihat jelas
    plt.xlim(0, 100) 
    plt.ylim(0, 105)
    
    # Menambahkan garis batas 95% energi (Target kompresi ideal)
    plt.axhline(95, color='gray', linestyle='--', alpha=0.8)
    plt.text(80, 96, 'Batas 95% Energi', color='black', fontsize=10)

    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")