
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. MEMBACA CITRA ASLI
# ==========================================

nama_file = "menwa.jpeg"

citra = cv2.imread(nama_file)

if citra is None:
    print("Gambar tidak ditemukan!")
    print("Periksa nama dan lokasi file gambar.")
    exit()

# Mengubah BGR menjadi RGB agar warna tampil benar
citra_rgb = cv2.cvtColor(citra, cv2.COLOR_BGR2RGB)


# ==========================================
# 2. MEAN FILTER
# ==========================================

mean_3x3 = cv2.blur(citra, (3, 3))
mean_5x5 = cv2.blur(citra, (5, 5))


# ==========================================
# 3. GAUSSIAN FILTER
# ==========================================

gaussian_3x3 = cv2.GaussianBlur(citra, (3, 3), 0)
gaussian_5x5 = cv2.GaussianBlur(citra, (5, 5), 0)


# ==========================================
# 4. MEDIAN FILTER
# ==========================================

median_3x3 = cv2.medianBlur(citra, 3)
median_5x5 = cv2.medianBlur(citra, 5)


# ==========================================
# 5. SHARPENING FILTER
# ==========================================

kernel_sharpening = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

sharpening = cv2.filter2D(
    citra, -1, kernel_sharpening
)


# ==========================================
# 6. DAFTAR HASIL DAN PENJELASAN
# ==========================================

judul = [
    "CITRA ASLI",
    "MEAN FILTER 3x3",
    "MEAN FILTER 5x5",
    "GAUSSIAN FILTER 3x3",
    "GAUSSIAN FILTER 5x5",
    "MEDIAN FILTER 3x3",
    "MEDIAN FILTER 5x5",
    "SHARPENING"
]

penjelasan = [
    "Gambar asli sebelum proses filtering.",
    "Mengurangi noise dengan rata-rata 3x3.",
    "Penghalusan lebih kuat menggunakan 5x5.",
    "Menghaluskan citra dengan Gaussian 3x3.",
    "Gaussian 5x5 memberikan blur lebih kuat.",
    "Mengurangi noise dengan nilai median 3x3.",
    "Menghilangkan noise lebih kuat dengan 5x5.",
    "Mempertegas tepi dan detail objek."
]

hasil = [
    citra_rgb,
    cv2.cvtColor(mean_3x3, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(mean_5x5, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(gaussian_3x3, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(gaussian_5x5, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(median_3x3, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(median_5x5, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(sharpening, cv2.COLOR_BGR2RGB)
]


# ==========================================
# 7. MENAMPILKAN HASIL FILTER
# ==========================================

fig, axes = plt.subplots(2, 4, figsize=(18, 10))

fig.suptitle(
    "PERBANDINGAN HASIL FILTER SPASIAL",
    fontsize=19,
    fontweight="bold",
    y=0.98
)

for i, ax in enumerate(axes.flat):

    ax.imshow(hasil[i])

    ax.set_title(
        judul[i],
        fontsize=12,
        fontweight="bold",
        pad=10
    )

    ax.text(
        0.5, -0.10,
        penjelasan[i],
        transform=ax.transAxes,
        fontsize=9,
        ha="center",
        va="top",
        wrap=True
    )

    ax.axis("off")

plt.subplots_adjust(
    left=0.03,
    right=0.97,
    top=0.88,
    bottom=0.12,
    wspace=0.12,
    hspace=0.35
)

# Menyimpan gambar hasil
plt.savefig(
    "hasil_filter_spasial_berwarna.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 8. ANALISIS NILAI INTENSITAS
# ==========================================

print("\n==========================================")
print("ANALISIS NILAI INTENSITAS CITRA")
print("==========================================")

daftar_hasil = {
    "Citra Asli": citra,
    "Mean Filter 3x3": mean_3x3,
    "Mean Filter 5x5": mean_5x5,
    "Gaussian Filter 3x3": gaussian_3x3,
    "Gaussian Filter 5x5": gaussian_5x5,
    "Median Filter 3x3": median_3x3,
    "Median Filter 5x5": median_5x5,
    "Sharpening": sharpening
}

for nama, gambar in daftar_hasil.items():

    # Mengubah ke grayscale untuk analisis intensitas
    gray = cv2.cvtColor(
        gambar, cv2.COLOR_BGR2GRAY
    )

    print("\n", nama)
    print("Intensitas minimum :", np.min(gray))
    print("Intensitas maksimum:", np.max(gray))
    print("Rata-rata intensitas:", round(np.mean(gray), 2))
    print("Standar deviasi     :", round(np.std(gray), 2))

print("\n==========================================")
print("PENGOLAHAN CITRA SELESAI")
print("==========================================")
print("File hasil: hasil_filter_spasial_berwarna.png")
