import cv2
import numpy as np
import glob
import math
import scipy
import time

def vectorToImg(v, row,col) :
    # Mengubah vektor v menjadi grid pixel gambar berukuran row x col
    # KAMUS LOKAL
    # i,j,ctr : integer
    # imgGrid, img : array of array of int
    # gridRow : array of int

    # ALGORITMA
    img = np.asarray(v.reshape(row,col), dtype=np.uint8)
    return img

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

def QREigenSendiri(mtrx, iteration=5000) :
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

def QREigenBuiltIn(mtrx, iteration=5000) :
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
        Q,R = np.linalg.qr(np.subtract(mK,smult)) # QR built-in
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

def rayleigh_iteration(mtrx):
    # Menghitung eigenvector dan eigenvalue memakai rayleigh quotient iteration
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

def InputFace(dir) :
    # Mengambil data wajah dari directory dir dan membuat matriks vektor baris gambar. Prekondisi : dir hanya berisi gambar.
    # KAMUS LOKAL

    # ALGORITMA
    initImage = np.array( [np.array(cv2.imread(image)) for image in glob.glob(f'{dir}/*')])
    trainingImage = np.array( [np.array(cv2.imread(image,0)) for image in glob.glob(f'{dir}/*')])
    imgVectorMatrix = []
    for images in trainingImage :
        # List penampung vektor gambar
        imgPixelList = images.flatten()     
        imgVectorMatrix.append(imgPixelList)
    return (imgVectorMatrix,initImage)

def MeanFace(imgVectorMatrix) :
    # Menghitung mean face dari matriks vektor gambar.
    # KAMUS LOKAL

    # ALGORITMA
    avgVector = np.zeros(len(imgVectorMatrix[0]), dtype=np.int64)
    count = 0
    for i in range(len(imgVectorMatrix)) :
        avgVector += imgVectorMatrix[i]
        count += 1
    for i in range(len(avgVector)) :
        avgVector[i] //= count
    avgVector = np.array(avgVector, dtype=np.uint8)
    return avgVector

def EigenFace(imgVectorMatrix, method) :
    # Melakukan kalkulasi eigenface terhadap matriks vektor baris gambar dan 
    # mengembalikan hasilnya, matriks koefisien terhadap gambar, waktu eksekusi, dan mean face 
    # sesuai metode eigen yang dipilih. 
    # KAMUS LOKAL

    # ALGORITMA 
    # Dataset kosong
    if (not(np.any(imgVectorMatrix))) :
        print("Tidak ada gambar di dataset")
        return
    
    # Mulai perhitungan waktu eksekusi
    start = time.time()
    # Selisih training image dengan nilai tengah (norm face)
    mean = MeanFace(imgVectorMatrix)
    for i in range(len(imgVectorMatrix)) :
        imgVectorMatrix[i] -= mean
    
    # Menghitung matriks kovarian
    mImgTrans = np.transpose(imgVectorMatrix)
    covar = np.dot(imgVectorMatrix, mImgTrans)

    # Menghitung nilai dan vektor eigen dari matriks kovarian dan mengurutkannya dari besar ke kecil. Diambil vektor yang signifikan saja.
    if (method=='QRBuiltIn') :
        eigValue, eigVector = QREigenBuiltIn(covar)
    else :
        if (method == 'QRSendiri') :
            eigValue, eigVector = QREigenSendiri(covar)
        else :
            if (method == 'Rayleigh') :
                eigValue, eigVector = rayleigh_iteration(covar)
    
    eigSortIdx = eigValue.argsort()[::-1]
    sorted_eigVal = eigValue[eigSortIdx]
    sort_eigVector = eigVector[:,eigSortIdx]
    largest_eigVector = []
    for i in range(len(sort_eigVector)) :
        if (i > 0) :
            if (abs(sorted_eigVal[i]/sorted_eigVal[i-1]) > 1e-3) :
                largest_eigVector.append(sort_eigVector[i])
            else :
                break
        else :
            largest_eigVector.append(sort_eigVector[i])
    largest_eigVector = np.array(largest_eigVector, dtype=np.float64).T
    print(len(largest_eigVector))

    # Membuat eigenface
    eigenFace = np.transpose(np.dot(mImgTrans, largest_eigVector))
    for i in range(len(eigenFace)) : # Normalisasi eigenface supaya hasil perhitungan jarak tidak terlalu besar
        eigenFace[i] /= np.linalg.norm(eigenFace[i])
    
    # Membuat vektor koefisien tiap gambar terhadap eigenFace
    coefTrain = []
    for i in range(len(imgVectorMatrix)) :
        coefI = []
        for j in range(len(eigenFace)) :
            coefI.append(np.dot(imgVectorMatrix[i],eigenFace[j]))
        coefTrain.append(coefI)
    
    end = time.time()
    execTime = end-start
    print("Waktu eksekusi : ", execTime)
    return mean, eigenFace, coefTrain, execTime

def RecognizeFace(dir, eigenFace, coefTrain, mean, initImage) :
    # Melakukan pengenalan wajah berdasarkan data eigenFace, mean, dan coefTrain yang ada
    # KAMUS LOKAL

    # ALGORITMA
    # Baca gambar uji
    testImg = cv2.imread(dir, 0).flatten()
    testImg -= mean

    # Projeksi ke eigenface
    coefTest = []
    for i in range(len(eigenFace)) :
        coefTest.append(np.dot(testImg,eigenFace[i]))
    idx = 0

    # Hitung kedekatan wajah dengan dataset berdasarkan Euclidean Distance
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
    if (min_dist > 1e4) :
        print("Wajah tidak ada di database")
    else :
        cv2.imwrite('closestImg.jpg', initImage[idx])

# Dicoba
imgVectorMtrx, initImage = InputFace('./split data/train/*')
mean, eigenFace, coefTrain, execTime = EigenFace(imgVectorMtrx, 'Rayleigh')
RecognizeFace('./testImg.jpg', eigenFace, coefTrain, mean, initImage)