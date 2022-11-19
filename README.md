# Tugas Besar 2 - Aljabar Linear dan Geometri
> Source Code ini dibuat oleh kami, Kelompok 11, untuk memenuhi Tugas Besar 2 - Aljabar Linear dan Geometri IF2123 yaitu mengimplementasikan 
> Aplikasi Nilai Eigen dan EigenFace pada Pengenalan Wajah (Face Recognition)

## Daftar Isi
* [Anggota Kelompok](#anggota-kelompok)
* [Implementasi Program](#implementasi-program)
* [Sistematika File](#sistematika-file)
* [Cara Menjalankan Program](#cara-menjalankan-program)

## Anggota Kelompok
NIM | Nama |
--- | --- |
13521108 | Michael Leon Putra Widhi |
13521145 | Kenneth Dave Bahana |
13521172 | Nathan Tenka

## Implementasi Program
Pada Tugas Besar kali ini, program yang kami buat dapat digunakan untuk :
1. Program melakukan pencocokan wajah dengan koleksi wajah yang ada di folder yang telah dipilih. Metrik untuk pengukuran kemiripan menggunakan eigenface + jarak euclidean.
2. Program menampilkan 1 hasil pencocokan pada dataset yang paling dekat dengan gambar input atau memberikan pesan jika tidak didapatkan hasil yang sesuai.
3. Program menghitung jarak euclidean dan nilai eigen & vektor eigen yang ditulis. Tidak boleh menggunakan fungsi yang sudah tersedia di dalam library atau Bahasa Python.
4. [BONUS] Terdapat fitur kamera yang dapat mengenali wajah secara realtime menggunakan webcam ketika program dijalankan dengan mekanisme pengambilan gambar yang dibebaskan.

## Sistematika File
```bash
.
├─── doc
├─── src
│   ├─── GUI
│   │   ├─── images
|   ├─── camRecord.py
|   ├─── cobaOpenCV.py
|   ├─── CobaQRDecomp.py
|   ├─── Eigenface.py
|   └─── webcam.py
├─── test
│   ├─── Dataset
│   ├─── Face Cam Data
│   └─── Gambar Uji
├─── Average face.jpg
├─── testImg.jpg
└─── README.md
```

## Cara Menjalankan Program
< isi disini >
