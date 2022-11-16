import numpy as np
import math
import scipy
import time
import splitfolders
import sympy

# Sumber QRDecomp lagi : https://pi.math.cornell.edu/~web6140/TopTenAlgorithms/QRalgorithm.html
# Sumber Lanczos algorithm : https://en.wikipedia.org/wiki/Lanczos_algorithm#Numerical_stability

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

def HHReflection(vec) :
    # Menghitung refleksi householder matriks mtrx
    # Sumber : https://scicomp.stackexchange.com/questions/30989/what-is-the-reason-that-lapack-uses-tau-in-qr-decomposition-instead-of-norma
    #          https://icl.utk.edu/lapack-forum/viewtopic.php%3Ff=2&t=5101.html#:~:text=Matlab%20use%20LAPACK%20to%20calculate%20the%20QR%20decomposition%2C,%27%20%22%3A%20transpose%20operator%20and%20I%3A%20identity%20matrix.
    # KAMUS LOKAL
    # ALGORITMA
    vecHH = np.copy(vec)
    n = vecHH.shape[0]
    alpha = vecHH[0]
    I = np.eye(n)
    if (n <= 1) :
        tau = 0
        return
    xnorm = np.linalg.norm(vecHH[1:])
    if (xnorm == 0) :
        tau = 0
    else : 
        beta = -np.sign(alpha) * math.sqrt(alpha**2 + xnorm**2)
        tau = (beta-alpha)/beta
        vecHH[1:] = vecHH[1:]/(alpha-beta)
        vecHH[0] = beta
    HH = I-tau*(vecHH @ vecHH.T)
    return HH

# def HessenbergReduction(mtrx) : # MASIH SALAH
#     # Menghasilkan matriks hessenberg dari mtrx
#     # Sumber : https://pi.math.cornell.edu/~web6140/TopTenAlgorithms/QRalgorithm.html
#     #          https://math.stackexchange.com/questions/732924/reducing-a-matrix-to-upper-hessenberg-form-using-householder-transformations-in
#     #          https://github.com/scipy/scipy/blob/v1.9.3/scipy/linalg/_decomp.py#L1359-L1436
#     # KAMUS LOKAL
#     # n : integer

#     # ALGORITMA
    
#     for i in range(n-2) :
#         return
#     transMtrx = mtrx.T
#     Hessenberg = mtrx
#     if (n > 2) :
#         a1 = transMtrx[0,1:]
#         e1 = np.zeros(n-1)
#         e1[0] = 1
#         sign = np.sign(a1[0])
#         v = (a1 + (sign*np.linalg.norm(a1)*e1)) # Pake norm built in dulu krn gtw knp kalo pake vectorLength overflow
#         v /= np.linalg.norm(v)
#         Q1 = np.eye(n-1) - 2*(v @ v.T)
#         mtrx[1:,0] = Q1 @ mtrx[1:,0]
#         mtrx[0,1:] = Q1 @ mtrx[0,1:]
#         mtrx[1:,1:] = Q1 @ mtrx[1:,1:] @ Q1.T
#         Hessenberg = HessenbergReduction(mtrx[1:,1:])
#     else :
#         Hessenberg = np.copy(mtrx)
#     return mtrx

def QRDecomp(mtrx) :
    # Memberikan hasil dekomposisi QR dari matriks mtrx
    # Sumber : https://www.ibm.com/docs/en/essl/6.2?topic=llss-sgeqrf-dgeqrf-cgeqrf-zgeqrf-general-matrix-qr-factorization
    #          https://www.r-bloggers.com/2017/04/qr-decomposition-with-householder-reflections/#:~:text=QR%20Decomposition%20with%20Householder%20Reflections%20The%20Householder%20reflection,A%20to%20construct%20the%20upper%20triangular%20matrix%20R.
    # KAMUS LOKAL
    # i,k : integer
    # Q, R : array of array of integer
    # u : array of integer
    
    # ALGORITMA
    k = min(len(mtrx),len(mtrx[0]))
    v = np.zeros(k)
    I = np.eye(k)
    Q = np.zeros(mtrx.shape)
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

def WilkinsonShift(a,b,c) :
    # Menghitung wilkinson shift untuk keperluan QR algorithm
    # KAMUS LOKAL
    # delta : float

    # ALGORITMA
    delta = (a-c)/2
    return c-(np.sign(delta)*pow(b,2)/(abs(delta)+math.sqrt(pow(delta,2)+pow(b,2))))

# def QR_EigValue(mtrx, iteration=10000) :
#     # Menghitung nilai eigen dari matrik mtrx memakai QR decomposition. Prekondisi : mtrx adalah matriks persegi
#     # Sumber : https://pi.math.cornell.edu/~web6140/TopTenAlgorithms/QRalgorithm.html
#     # KAMUS LOKAL 
#     # n : integer
#     # i : integer
    
#     # ALGORITMA
#     # Ditambahin cek waktu
#     startTime = time.time()
#     n = len(mtrx)
#     mK = scipy.linalg.hessenberg(mtrx)
#     mK = np.copy(mtrx)
#     eigVals = np.zeros(n)
#     if (n == 1) :
#         eigVals = mK[0,0]
#     else :
#         s = WilkinsonShift()
#         smult = np.eye(n) * s
#         # Q,R = QRDecomp(np.subtract(mK,smult))
#         # startQR = time.time()
#         Q,R = np.linalg.qr(np.subtract(mK,smult))
#         # endQR = time.time()
#         mK = np.add(R @ Q, smult)
#         QTdotQ = QTdotQ @ Q
#         if (i % 1000 == 0) :
#             print("Iterasi", i+1)
#         if (isUpperTriangular(mK)) :
#             break
#     # Waktu akhir
#     endTime = time.time()
#     print("Waktu eksekusi : ", endTime-startTime)
#     return np.diag(mK), QTdotQ

def QR_EigValue(mtrx, iteration=10000) :
    # Menghitung nilai eigen dari matrik mtrx memakai QR decomposition. Prekondisi : mtrx adalah matriks persegi
    # Sumber : https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition
    #          https://mathoverflow.net/questions/258847/solved-how-to-retrieve-eigenvectors-from-qr-algorithm-that-applies-shifts-and-d
    # KAMUS LOKAL 
    # n : integer
    # i : integerw
    
    # ALGORITMA
    # Ditambahin cek waktu
    startTime = time.time()
    n = len(mtrx)
    H,HQ = scipy.linalg.hessenberg(mtrx,calc_q=True)
    mK = H
    I = np.eye(n)
    QTdotQ = np.eye(n) # Matriks identitas ukuran n
    for i in range(iteration) :
        s = mK[n-1][n-1]
        smult = s * I
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
    print("QR execution time : ", endTime-startTime)
    return np.diag(mK), QTdotQ

# def QREigVector(mtrx, eigValues) :
#     # Mengembalikan eigenvector dari matriks persegi mtrx berdasarkan eigen value yang didapat
#     # KAMUS LOKAL
#     # i,j,k,n : integer
#     # echelon : array of array of integer
#     # currVector : array of float
#     # coefMtrx : array of array of float
#     # sum : float

#     # ALGORITMA
#     n = len(mtrx)
#     echelon = sympy.Matrix.rref(mtrx)
#     for eigVal in range(len(eigValues)) :
#         currVector = []
#         for k in range(n) :
#             currVector[k] = None
#         coefMtrx = np.eye(n)*eigVal - mtrx
#         for i in range(n-1,-1,-1) :
#             sum = 0
#             if (echelon[i].any) : # Baris tidak 0 semua
#                 for j in range(i+1,n) :
#                     sum += 

# def QRDecomp(mtrx) :
#     # Memberikan hasil dekomposisi QR dari matriks mtrx
#     # Sumber : https://www.codeproject.com/Articles/5319754/Can-QR-Decomposition-Be-Actually-Faster-Schwarz-Ru#mod_gs
#     # KAMUS LOKAL
#     # i, j, dotProduct : integer
#     # transM, Q, R, QTrans : array of array of integer
#     # u : array of integer

#     # ALGORITMA 
#     # transM = np.transpose(mtrx)
#     # QTrans = np.empty(transM.shape)
#     # R = np.empty((transM.shape[1], transM.shape[1]))
#     # u = np.empty(transM.shape[1])
#     # for i in range(len(transM)) :
#     #     for j in range(len(transM[i])) :
#     #         u[j] = float(transM[i][j])
#     #     if (i > 0) :
#     #         for j in range(i) :
#     #             dotProduct = np.dot(u, QTrans[j])
#     #             for k in range(len(u)) :
#     #                 u[k] -= QTrans[j][k]* dotProduct
#     #     lengthU = vectorLength(u)
#     #     if (lengthU != 0) :
#     #         for j in range (len(u)) :
#     #             u[j] /= lengthU
#     #     QTrans[i] = u
#     #     for j in range (len(R[i])) :
#     #         if (j >= i) :
#     #             R[i][j] = np.dot(u,transM[j])
#     #         else :
#     #             R[i][j] = 0
#     # Q = np.transpose(QTrans)
#     # return (Q,R)
#     Q = np.array(mtrx, dtype = float)
#     R = np.zeros((mtrx.shape[0], mtrx.shape[0]), dtype=float)
#     for k in range (len(R)) :
#         for i in range(k) :
#             R[i,k] = np.dot(Q[:,i], np.transpose(Q[:,k]))
#             Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]
#         R[k,k] = vectorLength(Q[:,k])
#         Q[:,k] = Q[:,k] / R[k,k]
#     return -Q,-R

# def QR_EigValue(mtrx, iteration=100000) :
#     # Menghitung nilai eigen dari matrik mtrx memakai QR decomposition. Prekondisi : mtrx adalah matriks persegi
#     # Sumber : https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition
#     # KAMUS LOKAL 
#     # n : integer
#     # i : integer
    
#     # ALGORITMA
#     n = len(mtrx)
#     mK = scipy.linalg.hessenberg(mtrx)
#     QTdotQ = np.eye(n) # Matriks identitas ukuran n
#     for i in range(iteration) :
#         s = WilkinsonShift(mK[n-2,n-2],mK[n-1,n-1],mK[n-2,n-1])
#         smult = np.eye(n) * s
#         Q,R = QRDecomp(np.subtract(mK,smult))
#         # Q,R = np.linalg.qr(np.subtract(mK,smult))
#         mK = np.add(R @ Q, smult)
#         QTdotQ = QTdotQ @ Q
#         if (i % 1000 == 0) :
#             print("Iterasi", i+1)
#         if (isUpperTriangular(mK)) :
#             break
#     return np.diag(mK), QTdotQ

# def power_iteration(mtrx, iteration=10000) :
#     v = np.random.randn(mtrx.shape[0])
#     v /= np.linalg.norm(v)
#     prevVec = np.copy(v)
#     for k in range(iteration) :
#         if (k % 1000 == 0) :
#             print("Iterasi ", k+1)
#         v = np.dot(mtrx,v)
#         v /= np.linalg.norm(v)
#         # if (np.allclose(v,prevVec)) :
#         #     break
#         # prevVec = v
#     return v

def rayleigh_iteration(mtrx):
    # Menghitung eigenvector dan eigenvalue memakai rayleigh quotient
    # Sumber : https://codereview.stackexchange.com/questions/229457/algorithm-that-generates-orthogonal-vectors-c-implementation (dapetin vektor ortogonal)
    #          https://en.wikipedia.org/wiki/Rayleigh_quotient_iteration
    startTime = time.time()
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
    endTime = time.time()
    print("Rayleigh execution time : ", endTime-startTime)
    return (eigValues, eigVectors.T)


# test = np.random.randint(0, 255, (5,5))
# test = np.dot(test, test.T)
# print(test)
# test = np.array([[ 56823, 68023, 85245, 91181, 37667],
#  [ 68023, 8820, 10390, 105639, 48333],
#  [ 8524, 10390, 17685, 16714, 105370],
#  [ 9118, 10563, 16714, 182378, 92210],
#  [ 37667, 4833, 105370, 92210, 76076]])
# print("Matriks awal")
# print(test)
# QSendiri, RSendiri = QRDecomp(test)
# QBuiltIn, RBuiltIn = np.linalg.qr(test)
# print("Sendiri")
# print(QSendiri)
# print(RSendiri)
# print(np.dot(QSendiri,RSendiri))
# print("Built in")
# print(QBuiltIn)
# print(RBuiltIn)
# print(np.dot(QBuiltIn,RBuiltIn))

# print("QR")
# M, QQ = QR_EigValue(test)
# print(M)
# print(QQ)

# # QR
# QRVal, QRVec = QR_EigValue(test)
# # Rayleigh
# value, vector = rayleigh_iteration(test)
# # Built in
# BuiltinValue, BuiltinVector = np.linalg.eig(test)

# eigSortIdxBuiltIn = BuiltinValue.argsort()[::-1] # argsort ngehasilin array yg isinya indeks elemen sesuai urutan.
# sorted_eigValBuiltIn = BuiltinValue[eigSortIdxBuiltIn]
# sorted_eigVectBuiltIn = BuiltinVector[:,eigSortIdxBuiltIn]

# eigSortIdxRayleigh = value.argsort()[::-1] # argsort ngehasilin array yg isinya indeks elemen sesuai urutan.
# sorted_eigValRayleigh = value[eigSortIdxRayleigh]
# sorted_eigVectRayleigh = vector[:,eigSortIdxRayleigh]

# eigSortIdxQR = QRVal.argsort()[::-1] # argsort ngehasilin array yg isinya indeks elemen sesuai urutan.
# sorted_eigValQR = QRVal[eigSortIdxQR]
# sorted_eigVectQR = QRVec[:,eigSortIdxQR]

# print("QR")
# print(sorted_eigValQR)
# print(sorted_eigVectQR)
# print()
# print("Rayleigh")
# print(sorted_eigValRayleigh)
# print(sorted_eigVectRayleigh)
# print()
# print("Built in")
# print(sorted_eigValBuiltIn)
# print(sorted_eigVectBuiltIn)
# Pake power iteration (salah)
# kLargestEigVector =[]
# for i in range (len(test)//10) :
#     print("Eigenvector ke",i+1)
#     kLargestEigVector.append(power_iteration(test))
# print(kLargestEigVector)

# Tes hessenberg reduction
# print("Matriks awal")
# print(test)
# print("Built in")
# print(scipy.linalg.hessenberg(test))
# print("Sendiri")
# print(HessenbergReduction(test))

# Tes Householder Reflection
test = np.array([[2,-2,18],[2,1,0],[1,2,0]])
HH = HHReflection(test[0])
print(HH)