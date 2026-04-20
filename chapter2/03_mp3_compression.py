import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

def main():
    # 1. PEMBUATAN AUDIO SINTETIS
    fs = 8000  # Frekuensi sampling 8 kHz
    t = np.linspace(0, 2.0, 2 * fs) # Durasi 2 detik
    
    # Nada dasar (440 Hz) + Nada tinggi (880 Hz) + Suara sirine/sweep (1000 ke 3000 Hz)
    audio_asli = 0.5 * np.sin(2 * np.pi * 440 * t) + \
                 0.3 * np.sin(2 * np.pi * 880 * t) + \
                 0.4 * signal.chirp(t, 1000, t[-1], 3000)
                 
    # Menambahkan sedikit noise
    audio_asli += 0.05 * np.random.randn(len(t))

    # 2. TRANSFORMASI KE DOMAIN WAKTU-FREKUENSI
    # nperseg = 256 adalah panjang blok (window size)
    f, t_spec, Zxx = signal.stft(audio_asli, fs, nperseg=256)

    # 3. PROSES KOMPRESI (PEMANGKASAN KOEFISIEN)
    Zxx_abs = np.abs(Zxx)
    
    # Mencari nilai batas (threshold) untuk membuang 90% data dengan energi terendah
    threshold = np.percentile(Zxx_abs, 90) 
    
    # Membuat filter: Simpan yang lebih besar dari threshold, buang sisanya
    mask = Zxx_abs >= threshold
    
    # Menerapkan pemangkasan ke matriks spektrum
    Zxx_compressed = Zxx * mask

    # Menghitung statistik kompresi
    total_coeffs = Zxx.size
    kept_coeffs = np.sum(mask)
    compression_ratio = (1 - (kept_coeffs / total_coeffs)) * 100

    # 4. REKONSTRUKSI (Inverse STFT)
    # Mengembalikan spektrum yang sudah dipotong kembali menjadi gelombang suara
    _, audio_rekonstruksi = signal.istft(Zxx_compressed, fs)
    
    # Menghitung kualitas (Signal-to-Noise Ratio)
    power_signal = np.mean(audio_asli**2)
    power_noise = np.mean((audio_asli - audio_rekonstruksi[:len(audio_asli)])**2)
    snr = 10 * np.log10(power_signal / power_noise)

    # 5. VISUALISASI HASIL
    plt.figure(figsize=(14, 8))

    # Spektrogram Asli
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t_spec, f, np.log10(Zxx_abs + 1e-10), shading='gouraud', cmap='viridis')
    plt.title('Spektrogram Audio Asli (100% Koefisien Disimpan)')
    plt.ylabel('Frekuensi (Hz)')
    plt.colorbar(label='Log Magnitudo')

    # Spektrogram Terkompresi
    plt.subplot(2, 1, 2)
    plt.pcolormesh(t_spec, f, np.log10(np.abs(Zxx_compressed) + 1e-10), shading='gouraud', cmap='viridis')
    plt.title(f'Spektrogram Kompresi MP3-style (Dibuang {compression_ratio:.1f}% Koefisien) | SNR: {snr:.1f} dB')
    plt.xlabel('Waktu (detik)')
    plt.ylabel('Frekuensi (Hz)')
    plt.colorbar(label='Log Magnitudo')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")