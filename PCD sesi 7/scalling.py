
import cv2
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread("kp.jpg")

# Memeriksa apakah gambar berhasil dibaca
if img is None:
    raise FileNotFoundError("Gambar kp.jpg tidak ditemukan!")

# Mengubah BGR menjadi RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Scaling dengan faktor 0,5 (memperkecil gambar)
hasil_05 = cv2.resize(
    img,
    None,
    fx=0.5,
    fy=0.5,
    interpolation=cv2.INTER_LINEAR
)

# Scaling dengan faktor 2 (memperbesar gambar)
hasil_2 = cv2.resize(
    img,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_LINEAR
)

# Menampilkan ukuran gambar
print("Ukuran gambar asli:", img.shape)
print("Ukuran scaling 0,5:", hasil_05.shape)
print("Ukuran scaling 2:", hasil_2.shape)

# Menampilkan hasil gambar
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Gambar Asli")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(hasil_05)
plt.title("Scaling 0,5 (Diperkecil)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(hasil_2)
plt.title("Scaling 2 (Diperbesar)")
plt.axis("off")

plt.tight_layout()
plt.show()
