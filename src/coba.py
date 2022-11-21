from tkinter import NW, Tk, Canvas, PhotoImage
import cv2 

def photo_image(img):
    h, w = img.shape[:2]
    data = f'P6 {w} {h} 255 '.encode() + img[..., ::-1].tobytes()
    return PhotoImage(width=w, height=h, data=data, format='PPM')

def update():
    ret, img = cap.read()
    if ret:
        photo = photo_image(img)
        canvas.create_image(0, 0, image=photo, anchor=NW)
        canvas.image = photo
    root.after(15, update)

root = Tk()
root.title("Video")
cap = cv2.VideoCapture(0)

canvas = Canvas(root, width=1200, height=700)
canvas.pack()
update()
root.mainloop()
cap.release()