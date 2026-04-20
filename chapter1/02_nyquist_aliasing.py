import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. ALIASING 1D (DOMAIN WAKTU / AUDIO)
    # Sinyal asli: Gelombang dengan frekuensi 7 Hz
    f_true = 7.0 
    
    t_cont = np.linspace(0, 1.0, 1000)
    y_cont = np.sin(2 * np.pi * f_true * t_cont)

    # Skenario Undersampling
    fs_under = 12.0
    t_under = np.arange(0, 1.0, 1/fs_under)
    y_under = np.sin(2 * np.pi * f_true * t_under)

    # Menghitung frekuensi palsu (Aliasing) yang terbentuk
    f_alias = abs(f_true - fs_under)
    y_alias = -np.sin(2 * np.pi * f_alias * t_cont) 

    # 2. ALIASING 2D (DOMAIN SPASIAL / CITRA)
    x = np.linspace(-1, 1, 400)
    y = np.linspace(-1, 1, 400)
    X, Y = np.meshgrid(x, y)
    
    # Membuat pola Radial Chirp
    R_squared = X**2 + Y**2
    img_high_res = np.sin(60 * np.pi * R_squared)

    # Mensimulasikan kamera beresolusi rendah dengan membuang piksel
    step = 4
    img_low_res = img_high_res[::step, ::step]

    # 3. VISUALISASI HASIL
    fig = plt.figure(figsize=(14, 8))

    # --- Plot 1D: Aliasing Sinyal ---
    ax1 = fig.add_subplot(2, 1, 1)
    # Plot sinyal asli (hijau)
    ax1.plot(t_cont, y_cont, 'g-', alpha=0.5, linewidth=2, label=f'Sinyal Asli ({f_true} Hz)')
    # Plot titik hasil sampling (merah)
    ax1.plot(t_under, y_under, 'ro', markersize=8, label=f'Sampel ({fs_under} Hz)')
    # Plot gelombang palsu yang menyambung titik merah (biru putus-putus)
    ax1.plot(t_cont, y_alias, 'b--', linewidth=2, label=f'Aliasing (Terbaca {f_alias} Hz)')
    
    ax1.set_title('Aliasing 1D: Undersampling Gelombang 7 Hz dengan Sampling Rate 12 Hz')
    ax1.set_xlabel('Waktu (detik)')
    ax1.set_ylabel('Amplitudo')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- Plot 2D: Moiré Pattern ---
    # Gambar Kiri: Resolusi Tinggi
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.imshow(img_high_res, cmap='gray', extent=[-1, 1, -1, 1])
    ax2.set_title('Gambar Asli (High-Res Radial Chirp)')
    ax2.axis('off')

    # Gambar Kanan: Resolusi Rendah (Undersampled)
    ax3 = fig.add_subplot(2, 2, 4)
    # Menampilkan gambar beresolusi rendah
    ax3.imshow(img_low_res, cmap='gray', extent=[-1, 1, -1, 1])
    ax3.set_title('Undersampled (Muncul Pola Moiré)')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")