from statistics import covariance
import cv2
import numpy as np
import os
import splitfolders
import glob

def transpose(mtrx) :
    # Menghasilkan transpose matris mtrx
    # KAMUS LOKAL
    # TMtrx : array of array of int
    # TArr : array of int
    # i,j : integer

    # ALGORITMA
    TMtrx = []
    for i in range (len(mtrx[0])) :
        TArr = []
        for j in range (len(mtrx)) :
            TArr.append(mtrx[j][i])
        TMtrx.append(TArr)
    return TMtrx

def multiplyMatrix(mtrx1,mtrx2) :
    # Mengalikan mtrx1 dan mtrx2
    # KAMUS LOKAL
    # mResult : array of array of int
    # i,j : integer
    # mulResult : integer

    # ALGORITMA
    mResult = [[0 for j in range(len(mtrx2[0]))] for i in range(len(mtrx1))]
    for i in range(len(mResult)) :
        for j in range(len(mResult[i])) :
            result = 0
            for k in range(len(mtrx2)) :
                result += mtrx1[i][k] * mtrx2[k][j]
            mResult[i][j] = result
    return mResult

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

# PROGRAM UTAMA
# KAMUS
# imgVectorMatrix, imgList : array of array of int

# Bagi gambar menjadi training dan test dataset
data_dir = "D:\OneDrive - Institut Teknologi Bandung\Folder Kuliah\Sem 3\Aljabar Linier dan Geometri\Tubes\Tubes 2\lfw-funneled"
main_dir = "D:\OneDrive - Institut Teknologi Bandung\Folder Kuliah\Sem 3\Aljabar Linier dan Geometri\Tubes\Tubes 2"
split_test_train(data_dir, "split data", 0.8)
output_dir = main_dir+"\split data"
# folder_dir = r"C:\Users\linal\OneDrive - Institut Teknologi Bandung\Folder Kuliah\Sem 3\Aljabar Linier dan Geometri\Tubes\Tubes 2\lfw-funneled\Angelina_Jolie"
imgVectorMatrix = []
avgVector = []
length = 0

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

for i in range(len(avgVector)) :
    avgVector[i] //= length
avgVector = np.asarray(avgVector, dtype=np.uint8)
# print(vectorToImg(avgVector,rows,cols))
cv2.imshow("average face",vectorToImg(avgVector,rows,cols))
cv2.waitKey(0) 

for i in range (len(imgVectorMatrix)) :
    imgVectorMatrix[i] -= avgVector

# print(imgVectorMatrix)
i = 1
for imgData in imgVectorMatrix :
    filename = "./norm face test/normFace"+str(i)+".jpg"
    cv2.imwrite(filename,vectorToImg(imgData,rows,cols))
    i += 1

# Matrix covariance
# mImgTrans = transpose(imgVectorMatrix)
# covariance = multiplyMatrix(imgVectorMatrix, mImgTrans)
# print(len(covariance))
# print(len(covariance[0]))