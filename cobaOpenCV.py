import cv2
import numpy as np
import splitfolders
import glob
import math
import scipy
import time
import sympy

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
    # Sumber : https://www.codeproject.com/Articles/5319754/Can-QR-Decomposition-Be-Actually-Faster-Schwarz-Ru#mod_gs
    # KAMUS LOKAL
    # i,k : integer
    # Q, R : array of array of integer
    # u : array of integer
    
    # ALGORITMA
    Q = np.array(mtrx, dtype = np.float64)
    R = np.zeros((mtrx.shape[0], mtrx.shape[0]), dtype=np.float64)
    for k in range (len(R)) :
        for i in range(k) :
            R[i,k] = np.dot(Q[:,i].T, Q[:,k])
            Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]
        R[k,k] = np.linalg.norm(Q[:,k])
        Q[:,k] = Q[:,k] / R[k,k]
    return -Q,-R

    # KAMUS LOKAL LAMA
    # i, j, dotProduct : integer
    # transM, Q, R : array of array of integer
    # u : array of integer
    
    # ALGORITMA LAMA
    # transM = np.transpose(mtrx)
    # QTrans = np.empty(transM.shape)
    # R = np.empty((transM.shape[1], transM.shape[1]))
    # u = np.empty(transM.shape[1])
    # for i in range(len(transM)) :
    #     for j in range(len(transM[i])) :
    #         u[j] = float(transM[i][j])
    #     if (i > 0) :
    #         for j in range(i) :
    #             dotProduct = np.dot(u, QTrans[j])
    #             for k in range(len(u)) :
    #                 u[k] -= QTrans[j][k]* dotProduct
    #     lengthU = vectorLength(u)
    #     if (lengthU != 0) :
    #         for j in range (len(u)) :
    #             u[j] /= lengthU
    #     QTrans[i] = u
    #     for j in range (len(R[i])) :
    #         if (j >= i) :
    #             R[i][j] = np.dot(u,transM[j])
    #         else :
    #             R[i][j] = 0
    # Q = np.transpose(QTrans)
    # return (Q,R)

def QR_EigValue(mtrx, iteration=5000) :
    # Menghitung nilai eigen dari matrik mtrx memakai QR decomposition. Prekondisi : mtrx adalah matriks persegi
    # Sumber : https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition
    #          https://mathoverflow.net/questions/258847/solved-how-to-retrieve-eigenvectors-from-qr-algorithm-that-applies-shifts-and-d
    # KAMUS LOKAL 
    # n : integer
    # i : integer
    
    # ALGORITMA
    # Ditambahin cek waktu
    startTime = time.time()
    n = len(mtrx)
    H, HQ = scipy.linalg.hessenberg(mtrx, calc_q=True)
    mK = H
    QTdotQ = np.eye(n) # Matriks identitas ukuran n
    for i in range(iteration) :
        s = mK[n-1][n-1]
        smult = np.eye(n) * s
        Q,R = QRDecomp(np.subtract(mK,smult))
        # startQR = time.time()
        # Q,R = np.linalg.qr(np.subtract(mK,smult)) # QR built-in
        # endQR = time.time()
        mK = np.add(R @ Q, smult)
        QTdotQ = QTdotQ @ Q
        if (i % 1000 == 0) :
            print("Iterasi", i+1)
        if (isUpperTriangular(mK)) :
            break
    QTdotQ = HQ @ QTdotQ
    # Waktu akhir
    endTime = time.time()
    print("Waktu eksekusi : ", endTime-startTime)
    return np.diag(mK), QTdotQ
    # n = len(mtrx)
    # mK = scipy.linalg.hessenberg(mtrx) 
    # mKPrev = np.copy(mK)
    # QTdotQ = np.eye(n) # Matriks identitas ukuran n
    # for i in range(iteration) :
    #     s = mK[n-1][n-1]
    #     smult = np.eye(n) * s
    #     Q,R = QRDecomp(np.subtract(mK,smult))
    #     mK = np.add(R @ Q, smult)
    #     QTdotQ = QTdotQ @ Q
    #     if (i % 1000 == 0) :
    #         print("Iterasi", i+1)
    #     if (isUpperTriangular(mK) and isDiagSame(mK, mKPrev)) :
    #         break
    #     mKPrev = np.copy(mK)
    # return np.diag(mK), QTdotQ

def rayleigh_iteration(mtrx):
    # Menghitung eigenvector dan eigenvalue memakai rayleigh quotient
    # Sumber : https://codereview.stackexchange.com/questions/229457/algorithm-that-generates-orthogonal-vectors-c-implementation (dapetin vektor ortogonal)
    #          https://en.wikipedia.org/wiki/Rayleigh_quotient_iteration
    n = mtrx.shape[0]
    max_iter = n
    eigVectors = []
    eigValues = np.empty(n)
    I = np.eye(n)
    v = np.random.randn(n) # Vektor tebakan acak
    v /= np.linalg.norm(v)
    for i in range(n) :
        v = np.random.randn(n) # Vektor tebakan acak
        if i > 0 :
            for k in range(len(eigVectors)) :
                v -= np.dot(v,eigVectors[k]) * eigVectors[k] # Cari vektor yang ortogonal dengan eigenvector sebelumnya
        v /= np.linalg.norm(v) # Normalisasi vektor
        mu = np.dot(v, np.dot(mtrx, v))
        for t in range(max_iter):
            try :
                v = np.linalg.inv(mu * I - mtrx) @ v # Selesaikan SPL (mu * I - mtrx) dengan v
                v /= np.linalg.norm(v)
                mu = np.dot(v, np.dot(mtrx, v)) # Hitung Rayleigh Quotient
            except :
                break
        eigValues[i] = mu
        eigVectors.append(v)
    eigVectors = np.array(eigVectors)
    return (eigValues, eigVectors.T)

# PROGRAM UTAMA
# KAMUS
# imgVectorMatrix, imgList : array of array of int

# ALGORITMA
# 0. Inisialisasi
startTime = time.time()
# Bagi gambar menjadi training dan test dataset
# Database awal
data_dir = "./Reduced face dataset"
split_test_train(data_dir, "split data", 0.8)
output_dir = "./split data"

# Database dengan image centered
# data_dir = "./Reduced face dataset centered"
# split_test_train(data_dir, "split data centered", 0.8)
# output_dir = "./split data centered"
# folder_dir = r"C:\Users\linal\OneDrive - Institut Teknologi Bandung\Folder Kuliah\Sem 3\Aljabar Linier dan Geometri\Tubes\Tubes 2\lfw-funneled\Angelina_Jolie"
imgVectorMatrix = []
avgVector = []
length = 0

# 1. Mengambil data wajah dari dataset
# Konversi ke grayscale dan menjadikannya matriks
# Wajah rata-rata dan bagi set training serta testing
initImage = np.array( [np.array(cv2.imread(image)) for image in glob.glob(f'{output_dir}/train/*/*')])
trainingImage = np.array( [np.array(cv2.imread(image,0)) for image in glob.glob(f'{output_dir}/train/*/*')])
# testImage = [cv2.imread(image,0) for image in glob.glob(f'{output_dir}/val/*/*')]

rows,cols = trainingImage[0].shape
print("Banyak gambar :", len(trainingImage))

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

# Munculin norm face
i = 1
for imgData in imgVectorMatrix :
    filename = "./norm face test/normFace"+str(i)+".jpg"
    cv2.imwrite(filename,vectorToImg(imgData,rows,cols))
    i += 1

# 4. Menghitung matriks kovarian
mImgTrans = np.transpose(imgVectorMatrix)
covar = np.dot(imgVectorMatrix, mImgTrans)

# 5. Menghitung nilai eigen dan vektor eigen (ternyata matriks kovariansnya simetris, jadi eigenvectornya adalah QQ)
eigValue, eigVector = rayleigh_iteration(covar) # Eigen sendiri
eigValueBuiltIn, eigVectorBuiltIn = np.linalg.eig(covar) # Eigen built in

# Urutkan nilai dan vektor eigen dari besar ke kecil
# Pake eigen built-in
eigSortIdxBuiltIn = eigValueBuiltIn.argsort()[::-1] # argsort ngehasilin array yg isinya indeks elemen sesuai urutan. BLH ATAU GA ?
sorted_eigValBuiltIn = eigValueBuiltIn[eigSortIdxBuiltIn]
sorted_eigVectBuiltIn = eigVectorBuiltIn[:,eigSortIdxBuiltIn]
sorted_eigVectorBuiltIn = []
for i in range(len(sorted_eigVectBuiltIn)//10) :
    sorted_eigVectorBuiltIn.append(sorted_eigVectBuiltIn[i])
# print(len(sorted_eigVector))
sorted_eigVectorBuiltIn = np.array(sorted_eigVectorBuiltIn, dtype=np.float64).T

# Eigen sendiri
eigSortIdx = eigValue.argsort()[::-1] # argsort ngehasilin array yg isinya indeks elemen sesuai urutan.
sorted_eigVal = eigValue[eigSortIdx]
sort_eigVector = eigVector[:,eigSortIdx]
sorted_eigVector = []
for i in range(len(sort_eigVector)//10) :
    sorted_eigVector.append(sort_eigVector[i])
# print(len(sorted_eigVector))
sorted_eigVector = np.array(sorted_eigVector, dtype=np.float64).T

for i in range(len(sorted_eigVal)) :
    if(abs(sorted_eigVal[i] - sorted_eigValBuiltIn[i]) > 1e-3) :
        print("Eigenvalue beda jauh")
# print(sorted_eigVal)
# print(sorted_eigValBuiltIn)

for i in range(len(sorted_eigVector)) :
    for j in range(len(sorted_eigVector[i])) :
        if (abs(abs(sorted_eigVector[i][j]) - abs(sorted_eigVectorBuiltIn[i][j])) > 1e-3) :
            print("Eigenvector beda jauh")
            break
# print(sorted_eigVector)
# print(sorted_eigVectorBuiltIn)

# print("Nilai eigen algo sendiri")
# print(eigValue)
# print(QQ)
# eigValueBuiltIn, eigVectorBuiltIn = np.linalg.eig(covar)
# print("Nilai eigen algo built in")
# print(eigValueBuiltIn)
# print(eigVectorBuiltIn)

# 6. Membuat eigenface
eigenFace = np.transpose(np.dot(mImgTrans, sorted_eigVector))
eigenFaceBuiltIn = np.transpose(np.dot(mImgTrans, sorted_eigVectorBuiltIn))
for i in range(len(eigenFace)) :
    eigenFace[i] /= np.linalg.norm(eigenFace[i])

# Bikin vektor koefisien tiap muka terhadap eigenFace
coefTrain = []
coefTrainBuiltIn = []
for i in range(len(imgVectorMatrix)) :
    coefI = []
    for j in range(len(eigenFace)) :
        coefI.append(np.dot(imgVectorMatrix[i],eigenFace[j]))
    coefTrain.append(coefI)

for i in range(len(imgVectorMatrix)) :
    coefI = []
    for j in range(len(eigenFaceBuiltIn)) :
        coefI.append(np.dot(imgVectorMatrix[i],eigenFaceBuiltIn[j]))
    coefTrainBuiltIn.append(coefI)

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

coefTestBuiltIn = []
for i in range(len(eigenFaceBuiltIn)) :
    coefTestBuiltIn.append(np.dot(testImg,eigenFaceBuiltIn[i]))

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
print(min_dist)
cv2.imwrite('closestImg.jpg', initImage[idx])

idx = 0
for i in range (len(coefTrainBuiltIn)) :
    distance = 0
    for j in range(len(coefTrainBuiltIn[i])) :
        if (j == 0) :
            distance = math.pow(coefTrainBuiltIn[i][j] - coefTestBuiltIn[j],2)
        else :
            distance += math.pow(coefTrainBuiltIn[i][j] - coefTestBuiltIn[j],2)
    distance = math.sqrt(distance)
    if (i == 0) :
        min_dist = distance
    else :
        if (distance < min_dist) :
            min_dist = distance
            idx = i
print(min_dist)
cv2.imwrite('closestImgBuiltIn.jpg', initImage[idx])

endTime = time.time()
print("Waktu eksekusi :", endTime-startTime)