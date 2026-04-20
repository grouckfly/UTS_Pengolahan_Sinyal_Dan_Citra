# Digital Signal and Image Processing (DSIP) 2026 - Simulasi Program

Repositori ini berisi kumpulan program simulasi menggunakan Python untuk mempelajari konsep dasar pengolahan sinyal dan citra, diadaptasi dari materi DSIP 2026. Setiap folder berisi implementasi kode yang memvisualisasikan teori matematika dan algoritma ke dalam bentuk grafik interaktif.

## Struktur Direktori

### chapter0/ - The Mathematical Language of Signals
Berisi program yang mendemonstrasikan fondasi matematika dalam pengolahan sinyal. Topik yang dicakup meliputi operasi matriks sebagai representasi sistem (blurring filter), analisis dinamika nilai eigen untuk stabilitas sistem, injeksi probabilitas noise (Gaussian, Uniform, Laplace) pada sinyal, serta teknik optimasi dan fenomena bias-variance tradeoff menggunakan Least Squares.

### chapter1/ - What is a Signal? The Unified Perspective
Berisi simulasi yang menjembatani dunia analog dan digital dengan sudut pandang universal. Program di dalamnya meliputi visualisasi persamaan gelombang ke dalam domain 1D (suara) dan 2D (citra), pembuktian teorema sampling Nyquist beserta fenomena aliasing (frekuensi palsu dan pola Moiré), serta simulasi kuantisasi sinyal (penurunan resolusi bit/stair-step).

### chapter2/ - The Dual Domains
Fokus pada representasi sinyal dalam dua dunia: domain waktu/ruang dan domain frekuensi. Folder ini disiapkan untuk simulasi yang membandingkan gelombang asli dengan spektrum frekuensinya menggunakan Transformasi Fourier, serta dasar-dasar kompresi seperti pemotongan frekuensi pada prinsip JPEG dan MP3.

### chapter3/ - Transformational Thinking
Berisi implementasi berbagai alat transformasi matematika untuk memecah sinyal menjadi komponen dasarnya. Topik meliputi analisis Fast Fourier Transform (FFT) dan efek windowing, Discrete Cosine Transform (DCT) yang menjadi tulang punggung algoritma kompresi, serta analisis Wavelet untuk sinyal transien.

### chapter4/ - Systems and Convolution
Berisi program yang mensimulasikan bagaimana sebuah sistem merespons dan memproses sinyal. Fokus utamanya adalah operasi konvolusi pada sinyal 1D dan 2D, karakterisasi sistem menggunakan Impulse Response atau Point Spread Function (PSF), serta perbandingan efisiensi komputasi filter.

---
**Catatan:** Pastikan telah menginstal library yang dibutuhkan (`numpy`, `scipy`, dan `matplotlib`) sebelum menjalankan program-program di dalam direktori ini.