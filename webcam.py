import cv2
import schedule

cam = cv2.VideoCapture(0)  # Index webcam, kebetulan main ku 1
img_counter = 0

def capture():
    global img_counter
    img_name = "image_{}.png".format(img_counter)
    cv2.imwrite(img_name, frame)
    print("Screenshot taken")
    img_counter += 1

# Set up schedule before loop
schedule.every(20).seconds.do(capture)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to take an image")
        break
    
    resize = cv2.resize(frame,(256,256))  # resize jadi 256 x 256
    cv2.imshow("Image Test", resize)
    schedule.run_pending()

    k = cv2.waitKey(100)  # 1/10 sec delay; no need for separate sleep

    if k % 256 == 27 or k == ord('q'):  # Command quit : q
        print("Closing the app")
        break

cam.release()  # Tutup cam
cv2.destroyAllWindows() # Hapus memmory cam agar tidak membebani laptop