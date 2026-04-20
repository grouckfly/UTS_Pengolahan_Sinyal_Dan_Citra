import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import warnings

warnings.filterwarnings('ignore')

# Fungsi bantuan untuk 2D DCT dan IDCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def main():
    # 1. PEMBUATAN CITRA SINTETIS
    N = 256
    x = np.linspace(-3, 3, N)
    y = np.linspace(-3, 3, N)
    X, Y = np.meshgrid(x, y)
    
    # Membuat gambar dengan variasi frekuensi (pola halus dan tekstur kasar)
    R = np.sqrt(X**2 + Y**2)
    img_original = np.sin(4 * R) + np.cos(3 * X) * np.sin(3 * Y) + np.exp(-(X**2 + Y**2)/2)
    # Normalisasi ke rentang piksel standar (0 - 255)
    img_original = (img_original - img_original.min()) / (img_original.max() - img_original.min()) * 255.0

    # 2. PERSIAPAN MASKING ZIGZAG (KUANTISASI)
    block_size = 8
    threshold = 4 
    mask = np.zeros((block_size, block_size))
    for i in range(block_size):
        for j in range(block_size):
            if i + j < threshold:
                mask[i, j] = 1

    # 3. PROSES KOMPRESI BLOK 8x8
    img_compressed = np.zeros_like(img_original)
    
    # Variabel untuk menyimpan 1 blok sebagai contoh analisis mendalam
    block_idx = N // 2
    sample_orig, sample_dct, sample_quant, sample_recon = None, None, None, None

    for i in range(0, N, block_size):
        for j in range(0, N, block_size):
            # Ambil blok 8x8
            block = img_original[i:i+block_size, j:j+block_size]
            
            # A. Transformasi DCT 2D
            dct_block = dct2(block)
            
            # B. Kuantisasi (Membuang frekuensi tinggi dengan mask)
            dct_quant = dct_block * mask
            
            # C. Inverse DCT 2D (Rekonstruksi kembali ke piksel)
            recon_block = idct2(dct_quant)
            img_compressed[i:i+block_size, j:j+block_size] = recon_block
            
            # Mengambil sampel blok di tengah gambar untuk diplot
            if i == block_idx and j == block_idx:
                sample_orig = block.copy()
                sample_dct = dct_block.copy()
                sample_quant = dct_quant.copy()
                sample_recon = recon_block.copy()

    # Menghitung kualitas (Peak Signal-to-Noise Ratio)
    mse = np.mean((img_original - img_compressed)**2)
    psnr = 10 * np.log10((255**2) / mse)

    # 4. VISUALISASI 1: HASIL KESELURUHAN
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title('Citra Asli (100% Data)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_compressed, cmap='gray')
    plt.title(f'Kompresi DCT (Hanya 15% Data)\nPSNR: {psnr:.2f} dB')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    error_img = img_original - img_compressed
    plt.imshow(error_img, cmap='coolwarm', vmin=-50, vmax=50)
    plt.title('Peta Error\n(Merah/Biru = Perubahan Besar)')
    plt.colorbar(label='Selisih Nilai Piksel')
    plt.axis('off')
    
    plt.tight_layout()

    # 5. VISUALISASI 2: ANALISIS MENDALAM 1 BLOK
    plt.figure(figsize=(12, 8))
    plt.suptitle('Analisis Dekomposisi 1 Blok 8x8', fontsize=16, fontweight='bold')

    plt.subplot(2, 3, 1)
    plt.imshow(sample_orig, cmap='gray', vmin=0, vmax=255)
    plt.title('1. Blok 8x8 Asli')

    plt.subplot(2, 3, 2)
    # Di-log agar titik energi terlihat jelas
    plt.imshow(np.log10(np.abs(sample_dct) + 1), cmap='magma')
    plt.title('2. Koefisien DCT\n(Pojok kiri atas = Frek. Rendah)')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('3. Filter Mask (Zigzag cutoff)\nPutih=Disimpan, Hitam=Dibuang')

    plt.subplot(2, 3, 4)
    plt.imshow(np.log10(np.abs(sample_quant) + 1), cmap='magma')
    plt.title('4. Kuantisasi\n(Sisa energi yang akan ditransmisikan)')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(sample_recon, cmap='gray', vmin=0, vmax=255)
    plt.title('5. Blok 8x8 Direkonstruksi')

    plt.subplot(2, 3, 6)
    plt.imshow(sample_orig - sample_recon, cmap='coolwarm', vmin=-30, vmax=30)
    plt.title('6. Error Rekonstruksi Blok')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJendela plot ditutup. Program selesai dijalankan.")