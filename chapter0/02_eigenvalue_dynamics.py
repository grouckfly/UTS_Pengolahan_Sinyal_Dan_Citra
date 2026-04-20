import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Definisikan Matriks Sistem (A)
    # Contoh 2: Sistem Tidak Stabil (Salah satu Eigenvalue > 1)
    A = np.array([[1.1, 0.2], [0.1, 0.9]])
    
    print("Matriks A:")
    print(A)

    # 2. Hitung Nilai Eigen dan Vektor Eigen
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("\nNilai Eigen (lambda):")
    print(eigenvalues)
    print("\nVektor Eigen (v):")
    print(eigenvectors)

    # Analisis stabilitas
    spectral_radius = np.max(np.abs(eigenvalues))
    print(f"\nSpectral Radius (max |lambda|): {spectral_radius:.4f}")
    if spectral_radius < 1:
        print("Kesimpulan: Sistem STABIL (konvergen ke 0)")
    elif spectral_radius > 1:
        print("Kesimpulan: Sistem TIDAK STABIL (divergen/membesar tanpa batas)")
    else:
        print("Kesimpulan: Sistem MARGINAL STABIL")

    # 3. Simulasi Dinamika Sistem (Iterasi)
    num_steps = 15
    x_history = np.zeros((2, num_steps))
    x_current = np.array([1.0, 1.0]) # Titik awal (x0)

    for i in range(num_steps):
        x_history[:, i] = x_current
        x_current = A @ x_current  # Update state: x(k+1) = A * x(k)

    # 4. Visualisasi Trajektori State
    plt.figure(figsize=(10, 5))

    # Plot 1: Pergerakan vektor di ruang state 2D
    plt.subplot(1, 2, 1)
    plt.plot(x_history[0, :], x_history[1, :], 'bo-', markersize=5, label='Trajektori x_k')
    plt.plot(x_history[0, 0], x_history[1, 0], 'go', markersize=8, label='Mulai (x_0)')
    
    # Gambar panah arah vektor eigen
    origin = [0, 0]
    plt.quiver(*origin, eigenvectors[0, 0], eigenvectors[1, 0], color='r', scale=3, label=f'v1 (λ={eigenvalues[0]:.2f})')
    plt.quiver(*origin, eigenvectors[0, 1], eigenvectors[1, 1], color='m', scale=3, label=f'v2 (λ={eigenvalues[1]:.2f})')

    plt.title('Trajektori Ruang State ($x_1$ vs $x_2$)')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axis('equal')

    # Plot 2: Perubahan nilai state terhadap waktu (iterasi)
    plt.subplot(1, 2, 2)
    plt.plot(range(num_steps), x_history[0, :], 'b-o', label='Komponen $x_1$')
    plt.plot(range(num_steps), x_history[1, :], 'r-o', label='Komponen $x_2$')
    plt.title('Evolusi State terhadap Waktu')
    plt.xlabel('Iterasi (k)')
    plt.ylabel('Nilai State')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")