import numpy as np
import matplotlib.pyplot as plt

def main():
    # Frekuensi yang sama untuk kedua domain: 5 siklus (gelombang)
    f = 5 

    # 1. Domain 1D: Sinyal Audio / Waktu
    # Membuat 500 titik waktu dari 0 sampai 1 detik
    t = np.linspace(0, 1, 500) 
    
    # Persamaan sinyal 1D (Amplitudo berfluktuasi antara -1 dan 1)
    signal_1d = np.sin(2 * np.pi * f * t)

    # 2. Domain 2D: Citra
    # Membuat grid koordinat X dan Y
    x = np.linspace(0, 1, 500)
    y = np.linspace(0, 1, 500)
    X, Y = np.meshgrid(x, y)

    # Persamaan sinyal 2D
    signal_2d = 0.5 + 0.5 * np.sin(2 * np.pi * f * X)

    # 3. Visualisasi Bersama
    fig = plt.figure(figsize=(12, 5))

    # Plot 1D (Kiri)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(t, signal_1d, 'b-', linewidth=2)
    ax1.set_title('Domain 1D: Sinyal Waktu (Gelombang Suara)')
    ax1.set_xlabel('Waktu (t)')
    ax1.set_ylabel('Amplitudo')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Plot 2D (Kanan)
    ax2 = fig.add_subplot(1, 2, 2)
    img = ax2.imshow(signal_2d, cmap='gray', extent=[0, 1, 0, 1], origin='lower')
    ax2.set_title('Domain 2D: Sinyal Spasial (Pola Citra)')
    ax2.set_xlabel('Ruang (x)')
    ax2.set_ylabel('Ruang (y)')
    fig.colorbar(img, ax=ax2, label='Intensitas Piksel')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")