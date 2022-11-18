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

def HouseHolder(vec) :
    # Menghasilkan reflektor householder berdasarkan vektor vec
    # Sumber : https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
    # KAMUS LOKAL

    # ALGORITMA
    normx = np.linalg.norm(vec)
    if (normx < 1e-5) :
        H = np.eye(len(vec))
        return H
    s = -np.sign(vec[0])
    u1 = vec[0] - s*normx
    w = vec/u1
    w[0] = 1
    tau = -s*u1/normx
    H = np.eye(len(vec)) - tau*w*np.reshape(w,(len(w),1))
    return H

def Tridiagonalize(mtrx) :
    # Ubah mtrx menjadi matriks tridiagonal
    # KAMUS LOKAL

    # ALGORITMA
    n = len(mtrx)
    m = np.array(mtrx,dtype=np.float64)
    Q = np.eye(n)
    if (n > 2) :
        for i in range(n-2) :
            HH = np.eye(n)
            HH[i+1:,i+1:] = HouseHolder(m[i+1:,i])
            m = HH @ m @ HH.T
            Q = HH @ Q
    for i in range(len(m)) :
        for j in range(len(m[i])) :
            if (abs(m[i][j]) < 1e-7) :
                m[i][j] = 0
    return m,Q

def QRDecomp(mtrx) :
    # Memberikan hasil dekomposisi QR dari matriks mtrx
    # Sumber : https://www.ibm.com/docs/en/essl/6.2?topic=llss-sgeqrf-dgeqrf-cgeqrf-zgeqrf-general-matrix-qr-factorization
    #          https://www.r-bloggers.com/2017/04/qr-decomposition-with-householder-reflections/#:~:text=QR%20Decomposition%20with%20Householder%20Reflections%20The%20Householder%20reflection,A%20to%20construct%20the%20upper%20triangular%20matrix%20R.
    #          https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
    # KAMUS LOKAL
    # i,k : integer
    # Q, R : array of array of integer
    # u : array of integer
    
    # ALGORITMA
    rows, cols = mtrx.shape
    Q = np.eye(rows)
    R = np.copy(mtrx)

    for i in range(cols) :
        HH = np.eye(cols)
        HH[i:,i:] = HouseHolder(R[i:,i])
        R = np.dot(HH,R)
        if (i<rows-1) :
            R[i+1:,i] = 0
        Q = np.dot(Q,HH)
    return Q,R

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
    H, HQ = Tridiagonalize(mtrx)
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
    print("Waktu eksekusi QR : ", endTime-startTime)
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
    # H, HQ = scipy.linalg.hessenberg(mtrx, calc_q=True)
    H, HQ = Tridiagonalize(mtrx)
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
    print("Waktu eksekusi QR : ", endTime-startTime)
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
    
    # Membuat eigenface dan mengurutkannya
    eigenFace = np.transpose(np.dot(mImgTrans, eigVector))
    for i in range(len(eigenFace)) : # Normalisasi eigenface supaya hasil perhitungan jarak tidak terlalu besar
        eigenFace[i] /= np.linalg.norm(eigenFace[i])

    eigSortIdx = eigValue.argsort()[::-1]
    sorted_eigVal = eigValue[eigSortIdx]
    sort_eigenFace = eigenFace[eigSortIdx]

    largest_eigenFace = []
    for i in range(len(sorted_eigVal)//10) :
        if (i > 0) :
            if (abs(sorted_eigVal[i]/sorted_eigVal[0]) > 1e-5) :
                largest_eigenFace.append(sort_eigenFace[i])
            else :
                break
        else :
            largest_eigenFace.append(sort_eigenFace[i])
    largest_eigenFace = np.array(largest_eigenFace, dtype=np.float64)

    # Membuat vektor koefisien tiap gambar terhadap eigenFace
    coefTrain = []
    for i in range(len(imgVectorMatrix)) :
        coefI = []
        for j in range(len(largest_eigenFace)) :
            coefI.append(np.dot(imgVectorMatrix[i],largest_eigenFace[j]))
        coefTrain.append(coefI)
    end = time.time()
    execTime = end-start
    print("Waktu eksekusi : ", execTime)
    return mean, largest_eigenFace, coefTrain, execTime

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
    avgDistance = 0
    for i in range (len(coefTrain)) :
        distance = 0
        for j in range(len(coefTrain[i])) :
            if (j == 0) :
                distance = math.pow(coefTrain[i][j] - coefTest[j],2)
            else :
                distance += math.pow(coefTrain[i][j] - coefTest[j],2)
        distance = math.sqrt(distance)
        avgDistance += distance
        if (i == 0) :
            min_dist = distance
        else :
            if (distance < min_dist) :
                min_dist = distance
                idx = i
    avgDistance /= len(coefTrain)
    print(avgDistance)
    print(min_dist)
    if (min_dist > 1e4) :
        print("Wajah tidak ada di database")
    else :
        cv2.imwrite('closestImg.jpg', initImage[idx])

# Dicoba
imgVectorMtrx, initImage = InputFace('./split data/train')
rows = len(initImage[0])
cols = len(initImage[0][0])
mean, eigenFace, coefTrain, execTime = EigenFace(imgVectorMtrx, 'QRBuiltIn')
decision = input()
while (decision != 'EXIT') :
    if (decision == 'R') :
        RecognizeFace('./testImg.jpg', eigenFace, coefTrain, mean, initImage)
    decision = input()