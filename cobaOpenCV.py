import cv2
import numpy as np
import glob

# Akses folder training dataset
# folder_dir = 'D:\OneDrive - Institut Teknologi Bandung\Folder Kuliah\Sem 3\Aljabar Linier dan Geometri\Tubes\Tubes 2\lfw-funneled\Angelina_Jolie'
# folder_dir = r"C:\Users\linal\OneDrive - Institut Teknologi Bandung\Folder Kuliah\Sem 3\Aljabar Linier dan Geometri\Tubes\Tubes 2\lfw-funneled\Angelina_Jolie"
folder_dir = r"..\Tubes2Algeo\Angelina_Jolie"
imgMatrix = []
avgVector = []
length = 0

imgList = [cv2.imread(image,0) for image in glob.glob(f'{folder_dir}/*')]
rows,cols = imgList[0].shape
for images in imgList :
    ctr = 0
    length += 1
    
    # List penampung nilai pixel
    imgPixelList = []
    
    for i in range(rows) :
        for j in range(cols) :
            imgPixelList.append(images[i][j])
            if (len(avgVector) < rows*cols) :
                avgVector.append(int(images[i][j]))
            else :
                avgVector[ctr] += int(images[i][j])
                ctr += 1
                
    imgMatrix.append(imgPixelList)
    
print(avgVector)

for i in range(len(avgVector)) :
    avgVector[i] //= length

print(avgVector)

ctr = 0
avg = []
for i in range(rows) :
    avgRow = []
    for j in range (cols) :
        avgRow.append(avgVector[ctr])
        ctr += 1
    avg.append(avgRow)
    
avgImage = np.asarray(avg, dtype=np.uint8)
cv2.imshow("average",avgImage)

for data in imgMatrix :
    for i in range(len(avgVector)) :
        data[i] -= avgVector[i]