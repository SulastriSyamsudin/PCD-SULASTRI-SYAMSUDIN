
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PROGRAM PERBAIKAN KECERAHAN FOTO
# Dengan Histogram Intensitas Piksel
# Nama file: Perbaikan.py
# ==========================================

BETA = 40

FOTO = [
    ("Foto 1 - Plat 2", "Plat 2.jpeg"),
    ("Foto 2 - Plat 3", "Plat 3.jpeg"),
]

# Folder untuk menyimpan semua hasil
OUTPUT_DIR = Path("hasil_kecerahan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rata_rata_kecerahan(gambar):
    """Menghitung rata-rata intensitas grayscale."""
    grayscale = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    return float(np.mean(grayscale))


def tingkatkan_kecerahan(gambar, beta):
    """
    Rumus peningkatan kecerahan:
    g(x,y) = min(255, max(0, f(x,y) + beta))
    """
    hasil = np.clip(
        gambar.astype(np.int16) + beta,
        0,
        255
    )
    return hasil.astype(np.uint8)


def buat_histogram(gambar):
    """Mengubah gambar menjadi grayscale untuk histogram."""
    return cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)


def main():
    data_foto = []

    # ==========================================
    # 1. MEMBACA DAN MEMPROSES FOTO
    # ==========================================
    for judul, nama_file in FOTO:
        path = Path(nama_file)

        if not path.exists():
            print(f"Foto tidak ditemukan: {nama_file}")
            print(
                "Pastikan foto berada satu folder "
                "dengan file Perbaikan.py."
            )
            return

        asli = cv2.imread(str(path))

        if asli is None:
            print(f"Foto gagal dibaca: {nama_file}")
            return

        # Meningkatkan kecerahan foto
        hasil = tingkatkan_kecerahan(asli, BETA)

        # Menghitung rata-rata kecerahan
        nilai_asli = rata_rata_kecerahan(asli)
        nilai_hasil = rata_rata_kecerahan(hasil)

        # Menyiapkan grayscale untuk histogram
        gray_asli = buat_histogram(asli)
        gray_hasil = buat_histogram(hasil)

        # Menyimpan foto hasil
        nama_output = OUTPUT_DIR / (
            path.stem.replace(" ", "_").lower()
            + "_terang.jpg"
        )

        cv2.imwrite(str(nama_output), hasil)

        data_foto.append({
            "judul": judul,
            "asli": cv2.cvtColor(
                asli, cv2.COLOR_BGR2RGB
            ),
            "hasil": cv2.cvtColor(
                hasil, cv2.COLOR_BGR2RGB
            ),
            "gray_asli": gray_asli,
            "gray_hasil": gray_hasil,
            "nilai_asli": nilai_asli,
            "nilai_hasil": nilai_hasil,
        })

        print(f"\n{judul}")
        print(f"Kecerahan asli  : {nilai_asli:.2f}/255")
        print(f"Kecerahan hasil : {nilai_hasil:.2f}/255")
        print(
            f"Perubahan       : "
            f"{nilai_hasil - nilai_asli:.2f}"
        )
        print(f"Foto disimpan   : {nama_output}")

    # ==========================================
    # 2. MENAMPILKAN FOTO DAN HISTOGRAM
    # ==========================================
    # Baris 1: Foto Plat 2
    # Baris 2: Foto Plat 3
    # Kolom 1: Foto asli
    # Kolom 2: Foto hasil
    # Kolom 3: Histogram asli dan hasil

    # ==========================================
    # 2. MENAMPILKAN FOTO DAN HISTOGRAM
    # ==========================================

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(20, 11),
        gridspec_kw={
            "width_ratios": [1, 1, 1.35],
            "height_ratios": [1, 1]
        }
    )
    
    fig.suptitle(
        "PERBANDINGAN FOTO DAN HISTOGRAM KECERAHAN",
        fontsize=17,
        fontweight="bold",
        y=0.97
    )



    for i, data in enumerate(data_foto):

        # --------------------------------------
        # FOTO ASLI
        # --------------------------------------
        axes[i, 0].imshow(data["asli"])
        axes[i, 0].set_title(
            f'{data["judul"]}\n'
            f'Foto Asli | {data["nilai_asli"]:.2f}/255',
            fontsize=11,
            pad=16
        )
        axes[i, 0].axis("off")

        # --------------------------------------
        # FOTO HASIL
        # --------------------------------------
        axes[i, 1].imshow(data["hasil"])
        axes[i, 1].set_title(
            f'{data["judul"]}\n'
            f'Hasil (+{BETA}) | {data["nilai_hasil"]:.2f}/255',
            fontsize=11,
            pad=16
        )
        axes[i, 1].axis("off")

        # --------------------------------------
        # HISTOGRAM
        # --------------------------------------
        ax = axes[i, 2]

        ax.hist(
            data["gray_asli"].ravel(),
            bins=256,
            range=(0, 256),
            color="royalblue",
            alpha=0.55,
            label="Foto asli"
        )

        ax.hist(
            data["gray_hasil"].ravel(),
            bins=256,
            range=(0, 256),
            color="tomato",
            alpha=0.50,
            label=f"Hasil (+{BETA})"
        )

        ax.set_title(
            f'Histogram - {data["judul"]}',
            fontsize=12,
            fontweight="bold",
            pad=10
        )

        ax.set_xlabel(
            "Intensitas Piksel (0-255)",
            fontsize=10,
            labelpad=7
        )
        ax.set_ylabel(
            "Jumlah Piksel",
            fontsize=10,
            labelpad=7
        )

        ax.set_xlim(0, 255)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(
            True,
            linestyle="--",
            alpha=0.25
        )
        ax.set_axisbelow(True)
        ax.legend(
            fontsize=9,
            loc="upper right",
            frameon=True
        )

    # Atur jarak agar judul dan grafik tidak bertumpuk
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.82,
        bottom=0.10,
        wspace=0.30,
        hspace=0.38
        )

    # --------------------------------------
    # SIMPAN HASIL PNG DAN PDF
    # --------------------------------------
    file_png = (
        OUTPUT_DIR / "perbandingan_kecerahan_histogram.png"
    )
    file_pdf = (
        OUTPUT_DIR / "perbandingan_kecerahan_histogram.pdf"
    )

    fig.savefig(
        file_png,
        dpi=250,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        file_pdf,
        bbox_inches="tight",
        facecolor="white"
    )

    print("\n===================================")
    print("PROSES SELESAI")
    print("===================================")
    print("Folder hasil :", OUTPUT_DIR.resolve())
    print("File PNG     :", file_png.resolve())
    print("File PDF     :", file_pdf.resolve())

    # Tampilkan hasil
    plt.show()
    
if __name__ == "__main__":
    print("Program mulai dijalankan...")
    main()

