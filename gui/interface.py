import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image

#window utama
window = tk.Tk()
window.geometry("1100x600")
window.configure(bg='#ffffff')
window.resizable(False,False)
window.title("Face Recognition")


#frame input (belum kepake)

input_frame = ttk.Frame(window)
input_frame.pack()

# placement Grid, Pack, Place
#functions
#def dataset():
    #popup.config = tk.Label(text="Test")
def insimg():
    #Insert image
    filename = filedialog.askopenfilename(filetypes=[("Image Files", '.png .jpeg')])
    print('Selected:', filename)
    #Display image

    theimg = Image.open(filename)
    resize_img = theimg.resize((256, 256), Image.Resampling.LANCZOS)
    img_input = ImageTk.PhotoImage(resize_img)

    #canvas.create_image(10, 10, anchor='nw', image=img)
    #canvas.pack()

    #img = ImageTk.PhotoImage(Image.open(filename));
    label = tk.Label(window, image = img_input)
    label.image = img_input
    label.place(relx = 0.3, rely = 0.35)
#main

# **** TITLE ******
#title - Face recognition
title_label = tk.Label(window, text="Face Recognition", font = ("Arial", 20), fg='#000000', bg='#ffffff')
title_label.place(relx=0.4, rely=0.1)

# **** INSERT COMPONENT ****
#label - Insert Your Dataset
label_dataset = tk.Label(window, text= "Insert Your Dataset", font=("Arial", 12), fg='#525252', bg='#ffffff')
label_dataset.place(relx=0.1, rely=0.35)


#choose file button - Insert Your Dataset
img_b = tk.PhotoImage(file="images/choose_file_button.PNG")
dataset_button = tk.Button(window, image=img_b, fg="blue", pady=20, padx=3, borderwidth=0, bg='#ffffff', activebackground='#ffffff')
dataset_button.place(relx = 0.095, rely = 0.4)
#label - Insert Your Image
label_insimage = tk.Label(window, text= "Insert Your Image", font=("Arial", 12), fg='#525252', bg='#ffffff')
label_insimage.place(relx=0.1, rely=0.5)
#choose file button - Insert Your Image
img_button = tk.Button(window, command= insimg, image=img_b, fg="blue",pady = 20, padx = 3, borderwidth=0, bg='#ffffff', activebackground='#ffffff')
img_button.place(relx = 0.095, rely = 0.55)

#Image background - title
label_testimg = tk.Label(window, text= "Test Image", bg='#ffffff', font=("Arial", 10), fg="#525252")
label_testimg.place(relx= 0.3, rely= 0.3)
#Image background
insbg = Image.open("images/imageless.PNG")
resize_insbg = insbg.resize((256, 256), Image.Resampling.LANCZOS)
img_inputless = ImageTk.PhotoImage(resize_insbg)

label_insbg = tk.Label(window, image=img_inputless, bg='#ffffff', borderwidth=0)
label_insbg.place(relx = 0.3, rely= 0.35)

#Closest Result background - title
label_testimg = tk.Label(window, text= "Closest Result", bg='#ffffff', font=("Arial", 10), fg="#525252")
label_testimg.place(relx= 0.6, rely= 0.3)
#Closest Result background

label_crbg = tk.Label(window, image=img_inputless, bg='#ffffff', borderwidth=0)
label_crbg.place(relx = 0.6, rely= 0.35)
window.mainloop()