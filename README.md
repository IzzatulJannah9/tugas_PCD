# Tugas 4 PCD — Filtering, Noise Reduction & Sharpening untuk OCR Nomor Ijazah

Membandingkan **Mean**, **Median**, **Gaussian filter** dan **Sharpening (unsharp mask)** sebagai tahap
pra-pemrosesan OCR (Tesseract) pada 9 variasi citra ijazah, plus eksperimen tambahan noise *salt-and-pepper*.

## Metode
1. Crop area `Nomor ijazah: 571012022000056`, ubah ke grayscale.
2. Enhancement: `cv2.blur` 3x3, `cv2.medianBlur` 3, `cv2.GaussianBlur` 3x3, unsharp mask (`addWeighted 1.5 / -0.5`, sigma 3).
3. Upscale 4x + Otsu threshold, lalu OCR Tesseract (`--psm 7`).
4. Akurasi = karakter digit benar berurutan (LCS) / 15 digit ground truth.

## Hasil (9 citra)
| Metode | Rata-rata akurasi (9 citra) | Min | Max |
|---|---|---|---|
| Original | 100.00% | 100.00% | 100.00% |
| Mean Filter | 97.04% | 93.33% | 100.00% |
| Median Filter | 91.11% | 73.33% | 100.00% |
| Gaussian Filter | 97.78% | 93.33% | 100.00% |
| Sharpening | 98.52% | 93.33% | 100.00% |

### Eksperimen salt-and-pepper (noise 8%)
| Metode | Hasil OCR | Akurasi |
|---|---|---|
| Noisy | (kosong) | 0.0% |
| Mean Filter | (kosong) | 0.0% |
| Median Filter | Nomor ijazah: 571012022000056 | 100.0% |
| Gaussian Filter | (kosong) | 0.0% |
| Median + Sharpening | Nomor ijazah: 571012022000056 | 100.0% |

Median filter memulihkan OCR dari 0% ke 100%, sedangkan Mean/Gaussian gagal karena merata-ratakan piksel ekstrem.
Pada citra yang sudah bersih, median filter justru bisa menurunkan akurasi (menghaluskan tepi karakter tipis) —
kualitas visual tidak selalu sejalan dengan akurasi OCR.

## Cara menjalankan
```bash
sudo apt install tesseract-ocr        # atau: brew install tesseract
pip install -r requirements.txt
# taruh image1.jpeg ... image9.jpeg di data/raw/
python src/pipeline.py --input data/raw --output results --gt 571012022000056
```
Gunakan `--rotate 0` jika citra sudah tegak.

## Struktur
```
src/pipeline.py   # seluruh pipeline
results/          # hasil filter (PNG) + results.json
report/           # laporan Word lengkap
data/raw/         # taruh citra sumber (tidak di-commit)
```

> **Privasi:** foto ijazah utuh berisi nama, tanggal lahir, foto, dan tanda tangan, sehingga dikecualikan lewat `.gitignore`.
