import cv2
import glob
import splitfolders

def split_test_train(inputDir, outputDir,x) :
    # Membagi file di inputDir menjadi training dan testing dataset dan memasukkannya ke folder outputDir
    # KAMUS LOKAL

    # ALGORITMA
    splitfolders.ratio(inputDir, output=outputDir, seed=1337, ratio = (x, 1-x))
    return 

# Ini buat resize
# i = 1
# for image in glob.glob(f'D:\OneDrive - Institut Teknologi Bandung\Folder Kuliah\Sem 3\Aljabar Linier dan Geometri\Tubes\Tubes 2\Dataset banyak/*/*') :
#     currImage = cv2.imread(image)
#     cv2.imwrite('D:\OneDrive - Institut Teknologi Bandung\Folder Kuliah\Sem 3\Aljabar Linier dan Geometri\Tubes\Tubes 2\Dataset banyak/Training'+str(i)+'.jpg',cv2.resize(currImage,(256,256)))
#     i += 1

# Ini buat ngebaginya, hasil yg dari resize hrs dimasukin ke satu folder dulu
# split_test_train('D:\OneDrive - Institut Teknologi Bandung\Folder Kuliah\Sem 3\Aljabar Linier dan Geometri\Tubes\Tubes 2\Dataset banyak','D:\OneDrive - Institut Teknologi Bandung\Folder Kuliah\Sem 3\Aljabar Linier dan Geometri\Tubes\Tubes 2\Dataset banyak',0.8)
