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

def HouseHolder(vec) :
    # Menghasilkan reflektor householder berdasarkan vektor vec
    # Sumber : https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
    # KAMUS LOKAL

    # ALGORITMA
    normx = np.linalg.norm(vec)
    if (normx == 0) :
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
            if (abs(i-j) > 1) :
                m[i][j] = 0
    return m,Q

def GivensRotation(a,b) :
    # Mencari matriks rotasi givens berukuran 2x2 
    # Sumber : https://en.wikipedia.org/wiki/Givens_rotation#Stable_calculation
    # KAMUS LOKAL

    # ALGORITMA
    # Kalkulasi nilai c,s dan r untuk givens rotation
    result = np.zeros((2,2),dtype=np.float64)
    if (b == 0) :
        c = np.sign(a)
        if (c == 0) :
            c = 1.0
        s = 0
        r = abs(a)
    elif (a == 0) :
        c = 0
        s = np.sign(b)
        r = abs(b)
    elif (abs(a) > abs(b)) :
        t = b/a
        u = np.sign(a)*math.sqrt(1+t*t)
        c = 1/u
        s = c*t
        r = a*u
    else :
        t = a/b
        u = np.sign(b)*math.sqrt(1+t*t)
        s = 1/u
        c = s*t
        r = b*u
    result[0][0] = c
    result[1][1] = c
    result[1][0] = -s
    result[0][1] = s
    return result

def QRDecompTridiag(mtrx) :
    # Menghasilkan dekomposisi QR dari matriks tridiagonal/Hessenberg atas mtrx
    # KAMUS LOKAL

    # ALGORITMA
    mK = np.array(mtrx,dtype=np.float64)
    n = len(mK)
    Q = np.eye(n,dtype=np.float64)
    for i in range(n-1) :
        givens = np.eye(n)
        givens[i:i+2,i:i+2] = GivensRotation(mK[i][i],mK[i+1][i])
        mK = givens @ mK
        mK[i+1][i] = 0
        Q = Q @ givens.T
    return Q,mK

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

    for i in range(cols-1) :
        HH = np.eye(cols)
        HH[i:,i:] = HouseHolder(R[i:,i])
        R = HH @ R
        if (i<rows-1) :
            R[i+1:,i] = 0
        Q = Q @ HH
    return Q,R

# def qr_gs_modsr(A, type=np.float64):
    
#     A = np.array(A, dtype=type)
    
#     (m,n) = np.shape(A) # Get matrix A's shape m - # of rows, m - # of columns
   
#     # Q - an orthogonal matrix of m-column vectors
#     # R - an upper triangular matrix (the Gaussian elimination of A to the row-echelon form)
    
#     # Initialization: [ Q - multivector Q = A of shape (m x n) ]
#     #                 [ R - multivector of shape (n x n)       ]

#     Q = np.array(A, dtype=type)      # Q - matrix A
#     R = np.zeros((n, n), dtype=type) # R - matrix of 0's    

#     # **** Objective: ****

#     # For each column vector r[k] in R:
#        # Compute r[k,i] element in R, k-th column q[k] in Q;

#     for k in range(n):
#         # For a span of the previous column vectors q[0..k] in Q, 
#         # compute the R[i,k] element in R as the inner product of vectors q[i] and q[k],
#         # compute k-th column vector q[k] as the product of scalar R[i,k] and i-th vector q[i],
#         # subtracting it from the k-th column vector q[k] in Q
#         for i in range(k):

#             # **** Compute k-th column q[k] of Q and k-th row r[k] of R **** 
#             R[i,k] = np.transpose(Q[:,i]).dot(Q[:,k])
#             Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]
            
#         # Compute the r[k,k] pseudo-diagonal element in R 
#         # as the Euclidean norm of the k-th vector q[k] in Q,

#         # Normalize the k-th vector q[k] in Q, dividing it by the norm r[k,k]
#         R[k,k] = np.linalg.norm(Q[:,k]); Q[:,k] = Q[:,k] / R[k,k]
    
#     return -Q, -R  # Return the resultant negative matrices Q and R 

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

def WilkinsonShift(a,b,c) :
    # Menghitung wilkinson shift untuk keperluan QR algorithm
    # KAMUS LOKAL
    # delta : float

    # ALGORITMA
    delta = (a-c)/2
    return c-(np.sign(delta)*pow(b,2)/(abs(delta)+math.sqrt(pow(delta,2)+pow(b,2))))

def QR_EigValue(mtrx, iteration=5000) :
    # Menghitung nilai eigen dari matrik mtrx memakai QR decomposition dengan Wilkinson Shift dan deflation. Prekondisi : mtrx adalah matriks persegi
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
    for i in range(n-1,0,-1) :
        iterQ = np.eye(i+1)
        while (abs(mK[i][i-1]) > 1e-5) :
            s = WilkinsonShift(mK[i-1][i-1],mK[i][i-1],mK[i][i])
            smult = np.eye(i+1) * s
            Q,R = QRDecompTridiag(np.subtract(mK[:i+1,:i+1],smult))
            # Q,R = np.linalg.qr(np.subtract(mK[:i+1,:i+1],smult)) # QR built-in
            mK[:i+1,:i+1] = np.add(R @ Q, smult)
            if (i < n-1) :
                mK[:i+1,i+1:] = Q.T @ mK[:i+1,i+1:]
            iterQ = iterQ @ Q
        paddedQ = np.eye(n)
        for k in range(len(iterQ)) :
            for j in range(len(iterQ)) :
                paddedQ[k][j] = iterQ[k][j]
        QTdotQ = QTdotQ @ paddedQ
    QTdotQ = HQ.T @ QTdotQ
    # Waktu akhir
    endTime = time.time()
    print("Waktu eksekusi QR : ", endTime-startTime)
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


test = np.random.randint(0, 255, (200,200))
test = np.dot(test, test.T)
print(test)
# Test tridiagonalisasi
# test = [[4,1,-2,2],[1,2,0,1],[-2,0,3,-2],[2,1,-2,-1]]
# print(test)
# tSendiri, HQ = Tridiagonalize(test)
# print(tSendiri)
# print(HQ.T @ tSendiri @ HQ)

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

# QR
QRVal, QRVec = QR_EigValue(test)
# Rayleigh
# value, vector = rayleigh_iteration(test)
# Built in
BuiltinValue, BuiltinVector = np.linalg.eig(test)

eigSortIdxBuiltIn = BuiltinValue.argsort()[::-1] # argsort ngehasilin array yg isinya indeks elemen sesuai urutan.
sorted_eigValBuiltIn = BuiltinValue[eigSortIdxBuiltIn]
sorted_eigVectBuiltIn = BuiltinVector[:,eigSortIdxBuiltIn]

# eigSortIdxRayleigh = value.argsort()[::-1] # argsort ngehasilin array yg isinya indeks elemen sesuai urutan.
# sorted_eigValRayleigh = value[eigSortIdxRayleigh]
# sorted_eigVectRayleigh = vector[:,eigSortIdxRayleigh]

eigSortIdxQR = QRVal.argsort()[::-1] # argsort ngehasilin array yg isinya indeks elemen sesuai urutan.
sorted_eigValQR = QRVal[eigSortIdxQR]
sorted_eigVectQR = QRVec[:,eigSortIdxQR]

print("QR")
print(sorted_eigValQR)
print(sorted_eigVectQR)
print()
# print("Rayleigh")
# print(sorted_eigValRayleigh)
# print(sorted_eigVectRayleigh)
# print()
print("Built in")
print(sorted_eigValBuiltIn)
print(sorted_eigVectBuiltIn)

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
# test = np.array([1,-2,2])
# HH = HouseHolder(test)
# print(HH)

# Test QR decomp Givens
# test, HQ = Tridiagonalize(test)
# startTime = time.time()
# Q,R = QRDecompTridiag(test)
# endTime = time.time()
# print(endTime-startTime)
# startTime = time.time()
# QBuilt,RBuilt = np.linalg.qr(test)
# endTime = time.time()
# print(endTime-startTime)
# print(Q,R)
# print(QBuilt,RBuilt)