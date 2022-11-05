import numpy as np
import math
import scipy

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
    mK = np.copy(mtrx)
    mKPrev = np.copy(mK)
    QTdotQ = np.eye(n) # Matriks identitas ukuran n
    for i in range(iteration) :
        s = mK[n-1][n-1]
        smult = np.eye(n) * s
        Q,R = QRDecomp(np.subtract(mK,smult))
        mK = np.add(R @ Q, smult)
        QTdotQ = QTdotQ @ Q
        if (i % 1000 == 0) :
            print("Iterasi", i+1)
        if (isLowerTriangular(mK) and isDiagSame(mK, mKPrev)) :
            break
        mKPrev = np.copy(mK)
    return np.diag(mK), QTdotQ

test = np.random.randint(0, 255, (5,5))
M, QQ = QR_EigValue(test)
print(M)
print(QQ)
print(np.linalg.eigvals(test))