import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

def main():
    # 1. PEMBUATAN SINYAL
    fs = 1000
    t = np.linspace(0, 2, 2 * fs)
    
    # Sinyal: Chirp linear (10Hz-300Hz) ditambah transien impuls mendadak
    chirp_sig = signal.chirp(t, f0=10, f1=300, t1=2, method='linear')
    transient = np.zeros_like(t)
    transient[int(1.0 * fs)] = 10 # Impuls tajam di detik ke-1.0
    
    signal_combined = chirp_sig + transient

    # 2. ANALISIS STFT (Jendela Tetap)
    f_stft, t_stft, Sxx = signal.stft(signal_combined, fs, nperseg=64)

    # 3. ANALISIS CWT (Wavelet Morlet & Ricker)
    widths = np.arange(1, 100)
    cwt_morlet, _ = pywt.cwt(signal_combined, widths, 'cmor1.5-1.0', sampling_period=1/fs)
    cwt_ricker, _ = pywt.cwt(signal_combined, widths, 'mexh', sampling_period=1/fs)

    # 4. VISUALISASI
    plt.figure(figsize=(14, 10))

    # Plot Sinyal Waktu
    plt.subplot(4, 1, 1)
    plt.plot(t, signal_combined, 'k', alpha=0.7)
    plt.title('Sinyal Waktu (Chirp + Transien Impuls)')
    plt.grid(True)

    # Plot STFT
    plt.subplot(4, 1, 2)
    plt.pcolormesh(t_stft, f_stft, np.abs(Sxx), shading='gouraud', cmap='viridis')
    plt.title('STFT Spectrogram (Fixed Resolution)')
    plt.ylabel('Frekuensi (Hz)')

    # Plot CWT Morlet
    plt.subplot(4, 1, 3)
    plt.imshow(np.abs(cwt_morlet), extent=[0, 2, 1, 100], cmap='viridis', aspect='auto', origin='lower')
    plt.title('CWT Morlet (Frequency-time localized)')
    plt.ylabel('Scale (Frequency)')

    # Plot CWT Ricker
    plt.subplot(4, 1, 4)
    plt.imshow(np.abs(cwt_ricker), extent=[0, 2, 1, 100], cmap='viridis', aspect='auto', origin='lower')
    plt.title('CWT Ricker (Transient-sharpness optimized)')
    plt.ylabel('Scale (Frequency)')
    plt.xlabel('Waktu (detik)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()