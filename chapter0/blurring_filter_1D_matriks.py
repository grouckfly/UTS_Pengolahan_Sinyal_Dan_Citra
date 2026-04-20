import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Membuat Sinyal Asli (x)
    n_samples = 15
    x = np.zeros(n_samples)
    x[5:10] = 1.0  # Sinyal bernilai 1 dari indeks 5 hingga 9
    print("Sinyal Asli (x):", x)

    # 2. Membuat Matriks Sistem (A) - Filter Blurring / Moving Average
    A = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        A[i, i] = 1/3
        if i > 0:
            A[i, i-1] = 1/3
        if i < n_samples - 1:
            A[i, i+1] = 1/3

    print("\nSebagian bentuk Matriks A (5x5 pertama):")
    print(np.round(A[:5, :5], 2))

    # 3. Menerapkan Sistem pada Sinyal (y = A * x)
    y = A @ x

    # 4. Visualisasi Hasil
    plt.figure(figsize=(10, 5))
    
    # Plot Sinyal Asli
    plt.subplot(1, 2, 1)
    plt.stem(range(n_samples), x, basefmt="b-", linefmt="b-", markerfmt="bo")
    plt.title('Sinyal Asli (x)')
    plt.xlabel('Indeks Waktu (n)')
    plt.ylabel('Amplitudo')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.2, 1.2)

    # Plot Sinyal Hasil Filter
    plt.subplot(1, 2, 2)
    plt.stem(range(n_samples), y, basefmt="r-", linefmt="r-", markerfmt="ro")
    plt.title('Sinyal Hasil Filter (y = Ax)')
    plt.xlabel('Indeks Waktu (n)')
    plt.ylabel('Amplitudo')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.2, 1.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()