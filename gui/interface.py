import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image
import time
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
    
    #Open Image
    theimg = Image.open(filename)
    #Resize Image
    resize_img = theimg.resize((256, 256), Image.Resampling.LANCZOS)
    #Framing Image
    img_input = ImageTk.PhotoImage(resize_img)
    #Creating image
    label = tk.Label(window, image = img_input)
    label.image = img_input
    #Display image
    label.place(relx = 0.3, rely = 0.35)

    #time
    #start = time.time()
    #minute = start//60
    #second = start%60
    #label_execution_time = tk.Label(text='f{minute}:{start}');
    #stop = time.time()


#main

# **** TITLE ******
#title - Face recognition
title_label = tk.Label(window, text="Face Recognition", font = ("Arial", 20), fg='#000000', bg='#ffffff')
title_label.place(relx=0.4, rely=0.1)

# Border Line Between Title & Other Components
#canvas
canvas = tk.Canvas(window, width = 910, height = 600, bg='#ffffff', borderwidth=0, border=0, highlightthickness=0)
canvas.place(relx=0.08, rely=0.2)

#elemen line
canvas.create_line(0,0,910,0, fill="black", width = 5);

# **** INSERT COMPONENT ****
#label - Insert Your Dataset
label_dataset = tk.Label(window, text= "Insert Your Dataset", font=("Arial", 12), fg='#525252', bg='#ffffff')
label_dataset.place(relx=0.1, rely=0.35)


#choose file button - Insert Your Dataset
img_b = tk.PhotoImage(file="images/choose_file_button.PNG")
dataset_button = tk.Button(window, image=img_b, fg="blue", pady=20, padx=3, borderwidth=0, bg='#ffffff', activebackground='#ffffff')
dataset_button.place(relx = 0.095, rely = 0.4)

#choosen file - Insert Your Dataset
dataset_info = tk.Label(window, text='No File Chosen', font=("Arial", 10),fg='#525252', bg='#ffffff')
dataset_info.place(relx = 0.173, rely = 0.416)
#label - Insert Your Image
label_insimage = tk.Label(window, text= "Insert Your Image", font=("Arial", 12), fg='#525252', bg='#ffffff')
label_insimage.place(relx=0.1, rely=0.5)
#choose file button - Insert Your Image
img_button = tk.Button(window, command= insimg, image=img_b, fg="blue",pady = 20, padx = 3, borderwidth=0, bg='#ffffff', activebackground='#ffffff')
img_button.place(relx = 0.095, rely = 0.55)
#choosen file - Insert Your Image
image_info = tk.Label(window, text='No File Chosen', font=("Arial", 10),fg='#525252', bg='#ffffff')
image_info.place(relx = 0.173, rely = 0.566)
#Image background - title
label_testimg = tk.Label(window, text= "Test Image", bg='#ffffff', font=("Arial", 10), fg="#525252")
label_testimg.place(relx= 0.3, rely= 0.3)

#Result - title
label_result_title = tk.Label(window, text='Result', font=("Arial", 12), fg='#525252', bg='#ffffff')
label_result_title.place(relx=0.1, rely=0.67)
#Result - the result
label_result_dir = tk.Label(window, text='None', font=("Arial", 10), fg='#5aff15', bg='#ffffff')
label_result_dir.place(relx=0.125, rely=0.72)
#Imageless image directory
insbg = Image.open("images/imageless.PNG")
resize_insbg = insbg.resize((256, 256), Image.Resampling.LANCZOS)
img_inputless = ImageTk.PhotoImage(resize_insbg)
#Image background
label_insbg = tk.Label(window, image=img_inputless, bg='#ffffff', borderwidth=0)
label_insbg.place(relx = 0.3, rely= 0.35)

#Closest Result background - title
label_testimg = tk.Label(window, text= "Closest Result", bg='#ffffff', font=("Arial", 10), fg="#525252")
label_testimg.place(relx= 0.6, rely= 0.3)
#Closest Result background

label_crbg = tk.Label(window, image=img_inputless, bg='#ffffff', borderwidth=0)
label_crbg.place(relx = 0.6, rely= 0.35)

#Execution time - title
label_execution_title = tk.Label(window, text="Execution time: ", font=("Arial", 9), fg="#525252", bg="#ffffff")
label_execution_title.place(relx = 0.3, rely = 0.8)
#Execution time - time
label_execution_time = tk.Label(window, text="00:00", font=("Arial", 9), bg='#ffffff', fg='#5aff15')
label_execution_time.place(relx=0.38, rely= 0.8)

window.mainloop()