import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. PEMBUATAN SINYAL ANALOG (KONTINU)
    fs = 1000  
    t = np.linspace(0, 1, fs)
    f = 2.0    
    
    # Sinyal asli dengan rentang amplitudo -1 hingga 1
    signal_analog = np.sin(2 * np.pi * f * t)

    # 2. PROSES KUANTISASI (Penurunan Resolusi)
    bit_depth = 5
    levels = 2 ** bit_depth  # 2^5 = 32 level
    
    print(f"Resolusi Bit: {bit_depth}-bit")
    print(f"Jumlah Level Tersedia: {levels} level")

    # Menggeser sinyal dari rentang [-1, 1] menjadi [0, 1] untuk perhitungan
    signal_normalized = (signal_analog + 1) / 2.0
    
    # Mengalikan dengan jumlah ruang level (levels - 1), lalu membulatkan ke bilangan bulat terdekat
    signal_quantized_int = np.round(signal_normalized * (levels - 1))
    
    # Mengembalikan sinyal yang sudah dibulatkan ke rentang amplitudo asli [-1, 1]
    signal_digital = (signal_quantized_int / (levels - 1)) * 2.0 - 1.0

    # 3. MENGHITUNG QUANTIZATION NOISE
    quantization_error = signal_digital - signal_analog

    # Menghitung Signal-to-Quantization-Noise Ratio (SQNR)
    power_signal = np.mean(signal_analog ** 2)
    power_noise = np.mean(quantization_error ** 2)
    sqnr_db = 10 * np.log10(power_signal / power_noise)
    print(f"SQNR (Kualitas Sinyal): {sqnr_db:.2f} dB")

    # 4. VISUALISASI HASIL
    fig = plt.figure(figsize=(14, 8))

    # --- Plot 1: Perbandingan Sinyal Analog dan Digital ---
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, signal_analog, 'g-', linewidth=3, alpha=0.5, label='Sinyal Analog (Mulus)')
    ax1.plot(t, signal_digital, 'b-', linewidth=1.5, label=f'Sinyal Digital ({bit_depth}-bit / {levels} Level)')
    ax1.set_title('Efek Kuantisasi: Pembentukan Pola Anak Tangga (Stair-step)')
    ax1.set_ylabel('Amplitudo')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- Plot 2: Zoom in untuk melihat efek "Stair-step" lebih jelas ---
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(t, signal_analog, 'g-', linewidth=3, alpha=0.5, label='Analog')
    ax2.plot(t, signal_digital, 'b-o', markersize=3, label='Digital')
    ax2.set_title('Zoom-in: Bentuk Anak Tangga')
    ax2.set_xlabel('Waktu (detik)')
    ax2.set_ylabel('Amplitudo')
    ax2.set_xlim(0.05, 0.20) # Zoom pada rentang waktu tertentu
    ax2.set_ylim(0.5, 1.05)  # Zoom pada puncak gelombang
    ax2.grid(True, linestyle=':', alpha=0.7)

    # --- Plot 3: Quantization Noise (Error) ---
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(t, quantization_error, 'r-', linewidth=1)
    ax3.set_title(f'Quantization Noise (Error Pembulatan)\nSQNR: {sqnr_db:.2f} dB')
    ax3.set_xlabel('Waktu (detik)')
    ax3.set_ylabel('Error Amplitudo')
    ax3.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")