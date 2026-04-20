import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal

def main():
    # 1. PERSIAPAN DATA
    N = 512
    img = np.random.rand(N, N)
    
    # Kernel Gaussian besar (21x21)
    K = 21
    k1d = signal.windows.gaussian(K, std=3).reshape(K, 1)
    kernel_2d = np.outer(k1d, k1d) # Membuat kernel 2D dari dua 1D
    kernel_2d /= kernel_2d.sum()

    print(f"Benchmarking Konvolusi pada Citra {N}x{N} dengan Kernel {K}x{K}...")
    print("-" * 60)

    # METODE 1: DIRECT CONVOLUTION
    start = time.time()
    res_direct = signal.convolve2d(img, kernel_2d, mode='same', boundary='fill')
    time_direct = time.time() - start
    print(f"1. Direct Convolution  : {time_direct:.4f} detik")

    # METODE 2: SEPARABLE CONVOLUTION
    start = time.time()
    # Konvolusi baris dulu, baru kolom
    row_conv = signal.convolve2d(img, k1d.T, mode='same')
    res_separable = signal.convolve2d(row_conv, k1d, mode='same')
    time_separable = time.time() - start
    print(f"2. Separable Kernels   : {time_separable:.4f} detik")

    # METODE 3: FFT-BASED CONVOLUTION
    start = time.time()
    res_fft = signal.fftconvolve(img, kernel_2d, mode='same')
    time_fft = time.time() - start
    print(f"3. FFT-based Conv      : {time_fft:.4f} detik")

    # VISUALISASI PERBANDINGAN WAKTU
    methods = ['Direct', 'Separable', 'FFT-based']
    times = [time_direct, time_separable, time_fft]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, times, color=['red', 'orange', 'green'])
    
    plt.title(f'Perbandingan Waktu Eksekusi (Citra {N}x{N}, Kernel {K}x{K})', fontsize=14)
    plt.ylabel('Waktu (detik)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Menambahkan label angka di atas bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}s', ha='center', va='bottom', fontweight='bold')

    # Menambahkan catatan speedup
    speedup = time_direct / time_fft
    plt.annotate(f'FFT {speedup:.1f}x lebih cepat\ndaripada Direct!', 
                 xy=(2, time_fft), xytext=(1.2, time_direct*0.7),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, color='darkgreen', fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram dihentikan.")