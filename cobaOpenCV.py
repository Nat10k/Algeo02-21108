import cv2
import numpy as np
import os
import splitfolders
import glob
import math
import scipy

# Prosedur Tambahan
def vectorToImg(v, row,col) :
    # Mengubah vektor v menjadi grid pixel gambar berukuran row x col
    # KAMUS LOKAL
    # i,j,ctr : integer
    # imgGrid, img : array of array of int
    # gridRow : array of int

    # ALGORITMA
    img = np.asarray(v.reshape(row,col), dtype=np.uint8)
    return img

def split_test_train(inputDir, outputDir,x) :
    # Membagi file di inputDir menjadi training dan testing dataset dan memasukkannya ke folder outputDir
    # KAMUS LOKAL

    # ALGORITMA
    splitfolders.ratio(inputDir, output=outputDir, seed=1337, ratio = (x, 1-x))
    return 

def vectorLength(v) :
    # Menghitung panjang vektor v
    # KAMUS LOKAL
    # length : integer
    # i : integer

    # ALGORITMA 
    length = 0
    for i in range(len(v)) :
        length += pow(v[i],2)
    length = math.sqrt(length)
    return length

def normalize(v) :
    # Menghasilkan vektor hasil normalisasi dari vektor v
    # KAMUS LOKAL
    # vNorm : array of int
    # vLength : int

    # ALGORITMA
    vLength = vectorLength(v)
    vNorm = v
    for i in range(len(vNorm)) :
        vNorm[i] /= vLength 
    return vNorm

def isUpperTriangular(mtrx) :
    # Mengembalikan True jika mtrx adalah matriks segitiga atas, False jika tidak
    # KAMUS LOKAL
    # i,j : integer

    # ALGORITMA 
    for i in range(1,len(mtrx)) :
        for j in range(i) :
            if(abs(mtrx[i][j]) > 1e-3) :
                return False
    return True

def isDiagSame(mtrx1, mtrx2) :
    # Mengembalikan True jika diagonal mtrx1 dan mtrx2 sama, False jika tidak
    # KAMUS LOKAL
    # i,j : integer

    # ALGORITMA 
    if (len(mtrx1) != len(mtrx2)) :
        return False
    for i in range(len(mtrx1)) :
        if (abs(mtrx1[i][i] - mtrx2[i][i]) > 1e-3) :
            return False
    return True

def QRDecomp(mtrx) :
    # Memberikan hasil dekomposisi QR dari matriks mtrx
    # KAMUS LOKAL
    # i, j, dotProduct : integer
    # transM, Q, R, QTrans : array of array of integer
    # u : array of integer

    # ALGORITMA 
    transM = np.transpose(mtrx)
    QTrans = np.empty(transM.shape)
    R = np.empty((transM.shape[1], transM.shape[1]))
    u = np.empty(transM.shape[1])
    for i in range(len(transM)) :
        for j in range(len(transM[i])) :
            u[j] = float(transM[i][j])
        if (i > 0) :
            for j in range(i) :
                dotProduct = np.dot(u, QTrans[j])
                for k in range(len(u)) :
                    u[k] -= QTrans[j][k]* dotProduct
        lengthU = vectorLength(u)
        if (lengthU != 0) :
            for j in range (len(u)) :
                u[j] /= lengthU
        QTrans[i] = u
        for j in range (len(R[i])) :
            if (j >= i) :
                R[i][j] = np.dot(u,transM[j])
            else :
                R[i][j] = 0
    Q = np.transpose(QTrans)
    return (Q,R)

def QR_EigValue(mtrx, iteration=100000) :
    # Menghitung nilai eigen dari matrik mtrx memakai QR decomposition. Prekondisi : mtrx adalah matriks persegi
    # Sumber : https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition
    # KAMUS LOKAL 
    # n : integer
    # i : integer
    
    # ALGORITMA
    n = len(mtrx)
    mK = scipy.linalg.hessenberg(mtrx) # sementara pake fungsi hessenberg built in (perlu implementasi sendiri)
    mKPrev = np.copy(mK)
    QTdotQ = np.eye(n) # Matriks identitas ukuran n
    for i in range(iteration) :
        Q,R = QRDecomp(mK)
        mK = R @ Q
        QTdotQ = QTdotQ @ Q
        if (i % 1000 == 0) :
            print("Iterasi", i+1)
        if (isUpperTriangular(mK) and isDiagSame(mK, mKPrev)) :
            break
        mKPrev = np.copy(mK)
    return np.diag(mK), QTdotQ

# PROGRAM UTAMA
# KAMUS
# imgVectorMatrix, imgList : array of array of int

# ALGORITMA
# 0. Inisialisasi
# Bagi gambar menjadi training dan test dataset
# data_dir = "./Reduced face dataset"
# split_test_train(data_dir, "split data", 0.8)
output_dir = "./split data"
# folder_dir = r"C:\Users\linal\OneDrive - Institut Teknologi Bandung\Folder Kuliah\Sem 3\Aljabar Linier dan Geometri\Tubes\Tubes 2\lfw-funneled\Angelina_Jolie"
imgVectorMatrix = []
avgVector = []
length = 0

# 1. Mengambil data wajah dari dataset
# Konversi ke grayscale dan menjadikannya matriks
# Wajah rata-rata dan bagi set training serta testing
trainingImage = np.array( [np.array(cv2.imread(image,0)) for image in glob.glob(f'{output_dir}/train/*/*')])
testImage = [cv2.imread(image,0) for image in glob.glob(f'{output_dir}/val/*/*')]

rows,cols = trainingImage[0].shape

for images in trainingImage :
    # List penampung nilai pixel
    imgPixelList = images.flatten()
    if (length == 0) :
        avgVector = imgPixelList.astype('int')
    else :
        avgVector += imgPixelList          
    imgVectorMatrix.append(imgPixelList)
    length += 1

# 2. Mengambil nilai tengah wajah
for i in range(len(avgVector)) :
    avgVector[i] //= length
avgVector = np.asarray(avgVector, dtype=np.uint8)
# print(vectorToImg(avgVector,rows,cols))

# Munculin mean face
# cv2.imshow("average face",vectorToImg(avgVector,rows,cols))
# cv2.waitKey(0) 

# 3. Selisih training image dengan nilai tengah
for i in range (len(imgVectorMatrix)) :
    imgVectorMatrix[i] -= avgVector

# print(imgVectorMatrix)
# i = 1

# Munculin norm face
# for imgData in imgVectorMatrix :
#     filename = "./norm face test/normFace"+str(i)+".jpg"
#     cv2.imwrite(filename,vectorToImg(imgData,rows,cols))
#     i += 1

# 4. Menghitung matriks covariance
mImgTrans = np.transpose(imgVectorMatrix)
covar = np.dot(imgVectorMatrix, mImgTrans)

# 5. Menghitung nilai eigen dan vektor eigen (ternyata matriks kovariansnya simetris, jadi eigenvectornya adalah QQ)
eigValue, QQ = QR_EigValue(covar)

# Urutkan nilai dan vektor eigen dari besar ke kecil
eigSortIdx = eigValue.argsort() # argsort ngehasilin array yg isinya indeks elemen sesuai urutan. BLH ATAU GA ?
sorted_eigVal = eigValue[eigSortIdx[:: -1]]
sorted_eigVector = QQ[eigSortIdx[:: -1]]
    
# print("Nilai eigen algo sendiri")
# print(eigValue)
# print(QQ)
# eigValueBuiltIn, eigVectorBuiltIn = np.linalg.eig(covar)
# print("Nilai eigen algo built in")
# print(eigValueBuiltIn)
# print(eigVectorBuiltIn)

# 6. Membuat eigenface
eigenFace = np.transpose(np.dot(mImgTrans, sorted_eigVector))
for i in range(len(eigenFace)) :
    eigenFace[i] = normalize(eigenFace[i])

# Bikin vektor koefisien tiap muka terhadap eigenFace
coefTrain = []
for i in range(len(imgVectorMatrix)) :
    coefI = []
    for j in range(len(eigenFace)) :
        coefI.append(np.dot(imgVectorMatrix[i],eigenFace[j]))
    coefTrain.append(coefI)

# Munculin eigen face (harus tanpa normalisasi (?))
# i = 1
# for eigFace in eigenFace :
#     filename = "./eigen face test/eigFace"+str(i)+".jpg"
#     cv2.imwrite(filename,vectorToImg(eigFace,rows,cols))
#     i += 1

# 7. Test pengenalan wajah
# Baca gambar uji
testImg = cv2.imread('./testImg.jpg', 0).flatten()
testImg -= avgVector
# cv2.imshow('Norm test image', vectorToImg(testImg,rows,cols))
# cv2.waitKey(0) 

# Projeksi ke eigenface space
coefTest = []
for i in range(len(eigenFace)) :
    coefTest.append(np.dot(testImg,eigenFace[i]))

# Coba rekonstruksi wajah test dari eigen face
# imgReconstruct = []
# for i in range(len(eigenFace)) :
#     if (i == 0) :
#         imgReconstruct = np.dot(eigCoef[i],eigenFace[i])
#     else :
#         imgReconstruct +=   np.dot(eigCoef[i],eigenFace[i])
# print(imgReconstruct)

# cv2.imwrite('reconstruct.jpg', vectorToImg(imgReconstruct, rows, cols))

# Cek pake Euclidean Distance
idx = 0
for i in range (len(coefTrain)) :
    distance = 0
    for j in range(len(coefTrain[i])) :
        if (j == 0) :
            distance = math.pow(coefTrain[i][j] - coefTest[j],2)
        else :
            distance += math.pow(coefTrain[i][j] - coefTest[j],2)
    distance = math.sqrt(distance)
    if (i == 0) :
        min_dist = distance
    else :
        if (distance < min_dist) :
            min_dist = distance
            idx = i

cv2.imwrite('closestImg.jpg', trainingImage[idx])