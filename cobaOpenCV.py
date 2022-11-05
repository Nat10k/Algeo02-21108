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

def isLowerTriangular(mtrx) :
    # Mengembalikan True jika mtrx adalah matriks segitiga bawah, False jika tidak
    # KAMUS LOKAL
    # i,j : integer

    # ALGORITMA 
    for i in range(1,len(mtrx)) :
        for j in range(i) :
            if(abs(mtrx[i][j]) > 1e-10) :
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
        if (abs(mtrx1[i][i] - mtrx2[i][i]) > 1e-10) :
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
        if (isLowerTriangular(mK) and isDiagSame(mK, mKPrev)) :
            break
        mKPrev = np.copy(mK)
    return np.diag(mK), QTdotQ

# PROGRAM UTAMA
# KAMUS
# imgVectorMatrix, imgList : array of array of int

# ALGORITMA
# 0. Inisialisasi
# Bagi gambar menjadi training dan test dataset
data_dir = "./Reduced face dataset"
split_test_train(data_dir, "split data", 0.8)
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
cv2.imshow("average face",vectorToImg(avgVector,rows,cols))
cv2.waitKey(0) 

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

# 4. Menghitung nilai covariance
mImgTrans = np.transpose(imgVectorMatrix)
covar = np.dot(imgVectorMatrix, mImgTrans)
print(covar)

# 5.a. Menghitung nilai eigen
eigValue, QQ = QR_EigValue(covar)
print("Nilai eigen algo sendiri")
print(eigValue)
eigValueBuiltIn = np.linalg.eigvals(covar)
print("Nilai eigen algo built in")
print(eigValueBuiltIn)

# 5.b. Menghitung vektor eigen