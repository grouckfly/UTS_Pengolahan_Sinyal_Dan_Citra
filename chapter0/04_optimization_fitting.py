import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():
    # 1. Pembuatan Data Sinyal
    np.random.seed(42)
    
    # Membuat 20 titik waktu (x) dari 0 sampai 5
    x = np.linspace(0, 5, 20)
    
    # Sinyal asli: Fungsi kubik (orde 3) -> y = 0.5x^3 - 2x^2 + x + 5
    y_true = 0.5 * x**3 - 2 * x**2 + x + 5
    
    # Menambahkan Gaussian noise ke sinyal asli
    noise = np.random.normal(0, 2.5, size=len(x))
    y_noisy = y_true + noise

    # x_plot untuk menggambar garis kurva mulus
    x_plot = np.linspace(-0.5, 5.5, 100)
    y_plot_true = 0.5 * x_plot**3 - 2 * x_plot**2 + x_plot + 5

    # 2. Proses Optimasi Least Squares (Curve Fitting)
    # Model 1: Orde 1 (Garis Lurus / Underfitting)
    coeffs_1 = np.polyfit(x, y_noisy, 1)
    y_fit_1 = np.polyval(coeffs_1, x_plot)

    # Model 2: Orde 3 (Sesuai dengan sinyal asli / Good Fit)
    coeffs_3 = np.polyfit(x, y_noisy, 3)
    y_fit_3 = np.polyval(coeffs_3, x_plot)

    # Model 3: Orde 12 (Terlalu kompleks / Overfitting)
    coeffs_12 = np.polyfit(x, y_noisy, 12)
    y_fit_12 = np.polyval(coeffs_12, x_plot)

    # 3. Visualisasi Hasil
    plt.figure(figsize=(15, 5))

    # Plot 1: Underfitting
    plt.subplot(1, 3, 1)
    plt.scatter(x, y_noisy, color='gray', label='Data + Noise')
    plt.plot(x_plot, y_plot_true, 'g-', label='Sinyal Asli (Target)', linewidth=2)
    plt.plot(x_plot, y_fit_1, 'r--', label='Model (Orde 1)', linewidth=2)
    plt.title('Underfitting (Terlalu Sederhana)')
    plt.xlabel('Waktu')
    plt.ylabel('Amplitudo')
    plt.ylim(-5, 20)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # Plot 2: Good Fit
    plt.subplot(1, 3, 2)
    plt.scatter(x, y_noisy, color='gray')
    plt.plot(x_plot, y_plot_true, 'g-', linewidth=2)
    plt.plot(x_plot, y_fit_3, 'r--', label='Model (Orde 3)', linewidth=2)
    plt.title('Good Fit (Optimasi Optimal)')
    plt.xlabel('Waktu')
    plt.ylim(-5, 20)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # Plot 3: Overfitting
    plt.subplot(1, 3, 3)
    plt.scatter(x, y_noisy, color='gray')
    plt.plot(x_plot, y_plot_true, 'g-', linewidth=2)
    plt.plot(x_plot, y_fit_12, 'r--', label='Model (Orde 12)', linewidth=2)
    plt.title('Overfitting (Menghafal Noise)')
    plt.xlabel('Waktu')
    plt.ylim(-5, 20)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")