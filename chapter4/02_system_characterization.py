import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def main():
    # 1. IMPULSE RESPONSE 1D
    # Membuat Impulse
    t = np.linspace(0, 1, 500)
    impulse_1d = np.zeros(500)
    impulse_1d[10] = 1.0
    
    # Membuat sistem (Filter Low-Pass)
    b, a = signal.butter(4, 0.05) # Filter Butterworth orde 4
    impulse_response = signal.lfilter(b, a, impulse_1d)

    # 2. POINT SPREAD FUNCTION 2D
    size = 100
    # Membuat 'Point Source' (Satu titik cahaya di tengah kegelapan)
    point_source = np.zeros((size, size))
    point_source[size//2, size//2] = 1.0
    
    # Membuat PSF (Model Gaussian Blur untuk mensimulasikan lensa yang tidak fokus)
    x = np.linspace(-3, 3, 21)
    y = np.linspace(-3, 3, 21)
    X, Y = np.meshgrid(x, y)
    psf_kernel = np.exp(-(X**2 + Y**2) / (2 * 1.0**2))
    psf_kernel /= psf_kernel.sum()
    
    # Output sistem optik (Hasil konvolusi titik dengan PSF)
    # Menggunakan mode='same' agar ukuran gambar tetap
    distorted_point = signal.convolve2d(point_source, psf_kernel, mode='same')

    # 3. VISUALISASI
    plt.figure(figsize=(14, 10))
    plt.suptitle('Karakterisasi Sistem: Impulse Response (1D) & PSF (2D)', fontsize=16)

    # Plot 1D: Input Impulse
    plt.subplot(2, 2, 1)
    plt.stem(t[:100], impulse_1d[:100], basefmt=" ")
    plt.title(r'Input: Unit Impulse ($\delta[n]$)')
    plt.ylabel('Amplitudo')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Plot 1D: Output (Impulse Response)
    plt.subplot(2, 2, 2)
    plt.plot(t[:100], impulse_response[:100], 'r', linewidth=2)
    plt.title('Output: Impulse Response ($h[n]$)\n(Karakteristik Sistem/Filter)')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Plot 2D: Point Source
    plt.subplot(2, 2, 3)
    plt.imshow(point_source, cmap='gray')
    plt.title('Input: Point Source (Ideal)')
    plt.axis('off')

    # Plot 2D: Point Spread Function (PSF)
    plt.subplot(2, 2, 4)
    # Menggunakan colormap 'hot' agar degradasi cahaya terlihat jelas
    plt.imshow(distorted_point, cmap='hot')
    plt.title('Output: Point Spread Function (PSF)\n(Karakteristik Lensa/Blur)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")