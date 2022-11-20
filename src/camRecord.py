import cv2
import time

faceDir = './test/Face_Cam_Data'
cam = cv2.VideoCapture(0)   # Index webcam, kebetulan main ku 1
userFace = input("Insert your name to record (use underscore as space): ")
print("Please wait the image record process")

start = time.time()
ambilData = 1

while True :  # Untuk menangkap gambar selama gaada kondisi berhenti
    retV, frame = cam.read()  # Capturing frame by frame
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert color to gray
    
    # Recording face data
    resize = cv2.resize(frame,(256,256))  # resize jadi 256 x 256
    namaFile = userFace+"_"+str(ambilData)+".jpg"
    cv2.imwrite(faceDir+'/'+namaFile,resize)
    ambilData += 1
    
    cv2.imshow('Recording Webcam',resize)   # Menampilkan gambar cam
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):   # Command quit : q
        break
    elif (ambilData > 30):  # Saat sudah terambil lebih dari 30 gambar
        break

finish = time.time()

print("\nFace record process completed!")
print("Total",ambilData-1,"faces recorded to database")
print("Time elapsed : ",finish-start,"second(s)")
cam.release()  # Tutup cam
cv2.destroyAllWindows()  # Hapus memmory cam agar tidak membebani laptop