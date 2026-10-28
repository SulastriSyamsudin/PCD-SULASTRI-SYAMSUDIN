
import imageio.v3 as imageio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Membaca gambar
nama_file = "gambar_input.jpg"
gambar = imageio.imread(nama_file)

print("Gambar berhasil dibaca!")
print("Ukuran gambar:", gambar.shape)

# 2. Mengubah gambar RGB menjadi grayscale
if gambar.ndim == 3:
    gambar_rgb = gambar[:, :, :3].astype(np.float32)

    grayscale = (
        0.299 * gambar_rgb[:, :, 0]
        + 0.587 * gambar_rgb[:, :, 1]
        + 0.114 * gambar_rgb[:, :, 2]
    ).astype(np.uint8)
else:
    grayscale = gambar.astype(np.uint8)

# 3. Menghaluskan gambar dengan Gaussian Filter
gambar_halus = ndimage.gaussian_filter(
    grayscale.astype(np.float32),
    sigma=1.5
)
gambar_halus = np.clip(gambar_halus, 0, 255).astype(np.uint8)

# 4. Mendeteksi tepi menggunakan Sobel Filter
sobel_x = ndimage.sobel(
    gambar_halus.astype(np.float32),
    axis=1
)

sobel_y = ndimage.sobel(
    gambar_halus.astype(np.float32),
    axis=0
)

tepi = np.hypot(sobel_x, sobel_y)

# Normalisasi hasil deteksi tepi
if tepi.max() > 0:
    tepi = (tepi / tepi.max() * 255).astype(np.uint8)
else:
    tepi = np.zeros_like(grayscale, dtype=np.uint8)

# 5. Menyimpan hasil pengolahan
folder_hasil = Path("hasil_pengolahan")
folder_hasil.mkdir(exist_ok=True)

imageio.imwrite(folder_hasil / "grayscale.png", grayscale)
imageio.imwrite(folder_hasil / "gambar_halus.png", gambar_halus)
imageio.imwrite(folder_hasil / "deteksi_tepi.png", tepi)

print("Semua hasil berhasil disimpan!")

# 6. Menampilkan perbandingan hasil
plt.figure(figsize=(14, 5))

plt.subplot(1, 4, 1)
plt.imshow(gambar)
plt.title("Gambar Asli")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(grayscale, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(gambar_halus, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(tepi, cmap="gray")
plt.title("Sobel Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    folder_hasil / "perbandingan_hasil.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
