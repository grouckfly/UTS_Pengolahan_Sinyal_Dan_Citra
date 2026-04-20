import numpy as np
import matplotlib.pyplot as plt

def convolve1d_manual(signal, kernel):
    """Implementasi manual sliding window untuk konvolusi 1D."""
    n_sig = len(signal)
    n_ker = len(kernel)
    pad_size = n_ker // 2
    padded_signal = np.pad(signal, pad_size, mode='edge')
    
    output = np.zeros(n_sig)
    # Proses Sliding Window
    for i in range(n_sig):
        # Ambil potongan sinyal seukuran kernel
        window = padded_signal[i : i + n_ker]
        # Kalikan dengan kernel (dibalik) dan jumlahkan
        output[i] = np.sum(window * kernel[::-1])
    return output

def convolve2d_manual(image, kernel):
    """Implementasi manual sliding window untuk konvolusi 2D."""
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    # Padding gambar
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros((i_h, i_w))
    
    # Proses Sliding Window (2D)
    for i in range(i_h):
        for j in range(i_w):
            # Ambil blok piksel seukuran kernel
            window = padded_img[i : i + k_h, j : j + k_w]
            # Dot product
            output[i, j] = np.sum(window * np.flip(kernel))
    return output

def main():
    # 1. SIMULASI 1D: SMOOTHING (PERATAAN)
    t = np.linspace(0, 1, 200)
    # Sinyal kotak dengan banyak noise
    clean_signal = np.where((t > 0.3) & (t < 0.7), 1.0, 0.0)
    noisy_signal = clean_signal + np.random.normal(0, 0.2, len(t))
    
    # Kernel Moving Average (smoothing)
    kernel_1d = np.ones(15) / 15
    smoothed_signal = convolve1d_manual(noisy_signal, kernel_1d)

    # 2. SIMULASI 2D: EDGE DETECTION (SOBEL)
    # Membuat gambar kotak sederhana
    img = np.zeros((100, 100))
    img[25:75, 25:75] = 1.0
    
    # Kernel Sobel (Horizontal) untuk deteksi tepi tajam
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    edges = convolve2d_manual(img, sobel_x)

    # 3. VISUALISASI
    plt.figure(figsize=(14, 8))
    plt.suptitle('Manual Convolution: 1D Smoothing & 2D Edge Detection', fontsize=16)

    # Plot 1D
    plt.subplot(2, 2, 1)
    plt.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy Input')
    plt.plot(t, smoothed_signal, 'r', linewidth=2, label='Smoothed Output')
    plt.title('1D Convolution (Smoothing)')
    plt.legend()

    # Plot 2D Asli
    plt.subplot(2, 2, 3)
    plt.imshow(img, cmap='gray')
    plt.title('2D Input (Original Image)')
    plt.axis('off')

    # Plot 2D Tepi
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(edges), cmap='magma')
    plt.title('2D Convolution (Sobel Edge Detection)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")