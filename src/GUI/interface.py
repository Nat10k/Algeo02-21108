import tkinter as tk
from tkinter import  filedialog
from PIL import ImageTk, Image
import time
#window utama
window = tk.Tk()
window.geometry("1100x600")
window.configure(bg='#000000')
window.attributes('-topmost')
#window.wm_attributes('-transparentcolor', '#ffffff')
window.resizable(False,False)
window.title("Face Recognition")
#window.wm_attributes('-transparentcolor', '#ffffff')
#window.wm_attributes('-alpha',0.5)
#background window
#C = tk.Canvas(window, bg="blue", height=250, width=300)
#window_bg_img = tk.PhotoImage(file='./gui/images/windowbackground.png')
#window_bg_img_open = Image.open("./gui/images/windowbackground.png")
#window_bg_img = tk.PhotoImage(file='images/orangething.png')
#window_bg_img_open = Image.open("images/orangething.png")
#resize_window_bg_img = window_bg_img_open.resize((1100, 600), Image.Resampling.LANCZOS)
#window_background = ImageTk.PhotoImage(resize_window_bg_img)
#window_bg_label = tk.Label(window, image=window_background)
#window_bg_label.image = window_background

#window_bg_label.place(x=0, y=0)
#window_bg_label.pack()



# placement Grid, Pack, Place
#functions
#def dataset():
    #popup.config = tk.Label(text="Test")

#def closestresult():
    

def insimg(label_chosen):
    #Insert image
    filename = filedialog.askopenfilename(filetypes=[("Image Files", '.png .jpeg')])
    print('Selected:', filename)
    
    #Chosen file label
    label_chosen = tk.Label(window, text=filename, font=("Arial", 8),fg='#525252', bg='#ffffff', wraplength=135, justify='left')
    label_chosen.place(relx = 0.173, rely = 0.575)
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
    label.place(relx = 0.33, rely = 0.35)

def dataimg(label_1):
    file = filedialog.askdirectory(parent=window, title='Open 1st file')
    print('Dataset: ', file)
    label_1 = tk.Label(window, text= file, wraplength=135, font=("Arial", 8),fg='#525252', bg='#ffffff', justify='left')
    label_1.place(relx = 0.173, rely = 0.400)
    #time
    #start = time.time()
    #minute = start//60
    #second = start%60
    #label_execution_time = tk.Label(text='f{minute}:{start}');
    #stop = time.time()


#main
transparent = tk.PhotoImage(file='src/GUI/images/transparent.png')
# **** TITLE ******
#title - Face recognition
title_label = tk.Label(window, text="Face Recognition", font = ("Lucida Handwriting", 20, "bold"), fg='#ffffff', bg='#100000')
title_label.place(relx=0.37, rely=0.1)

# Border Line Between Title & Other Components
#canvas
canvas = tk.Canvas(window, width = 910, height = 3, bg='#ffffff', borderwidth=0, border=0, highlightthickness=0)
canvas.place(relx=0.08, rely=0.2)

#elemen line
canvas.create_line(0,0,910,0, fill="black", width = 5)
# Orange Canvas
orangebg_canvas = tk.Canvas(window, width=1094, height=475, bg='#FF5733', highlightthickness=3)
orangebg_canvas.place(relx=0, rely=0.2)
# **** INSERT COMPONENT ****
#label - Insert Your Dataset

left_canvas = tk.Canvas(window, width=235, height=400, borderwidth=5, border=5, highlightthickness =4,highlightbackground="black", bg="#ffffff")
left_canvas.place(relx=0.075, rely = 0.27)
#left_bg_img = Image.open("images/softcolorbg.png")
#leftbgimg = left_bg_img.resize((250, 400))
#lbi = ImageTk.PhotoImage(leftbgimg)
#left_canvas.create_image(125, 200, image=lbi)


insds_canvas = tk.Canvas(window, width=235, height=20, borderwidth=5, border=5, bg="#FFAA33", highlightbackground="black", highlightthickness=4)
insds_canvas.place(relx=0.075, rely=0.325)
label_dataset = tk.Label(window, text= "Insert Your Dataset", font=("Papyrus", 11, 'bold'), fg='#000000', bg="#FFAA33")
label_dataset.place(relx=0.1, rely=0.331)
#choose file button - Insert Your Dataset
#img_b = tk.PhotoImage(file="./gui/images/choose_file_button.PNG")
img_b = tk.PhotoImage(file="src/GUI/images/choose_file_button.PNG")
dataset_button = tk.Button(window, command= lambda:dataimg(dataset_info), image=img_b, fg="blue", pady=20, padx=3, borderwidth=0, border=0, highlightthickness=0,bg='#ffffff', activebackground='#ffffff')
dataset_button.configure(fg ="white")
dataset_button.place(relx = 0.095, rely = 0.4)

#choosen file - Insert Your Dataset
dataset_info = tk.Label(window, text='No File Chosen', font=("Arial", 10),fg='#525252', bg='#ffffff')
dataset_info.place(relx = 0.175, rely = 0.416)

#label - Insert Your Image
insimg_canvas = tk.Canvas(window, width=235, height=20, borderwidth=5, border=5, bg="#FFAA33", highlightbackground="black", highlightthickness=4)
insimg_canvas.place(relx=0.075, rely=0.51)
label_insimage = tk.Label(window, text= "Insert Your Image", font=("Papyrus", 11, 'bold'), fg='#000000', bg='#FFAA33')
label_insimage.place(relx=0.1, rely=0.516)

#choose file button - Insert Your Image
img_button = tk.Button(window, command= lambda:insimg(image_info), image=img_b, fg="blue",pady = 20, padx = 3, borderwidth=0, bg='#ffffff', activebackground='#ffffff')
img_button.place(relx = 0.095, rely = 0.58)

#choosen file - Insert Your Image
image_info = tk.Label(window, text='No File Chosen', font=("Arial", 10),fg='#525252', bg='#ffffff')
image_info.place(relx = 0.175, rely = 0.596)

#Image background - title
testimg_canvas = tk.Canvas(window, width=74, height=27, borderwidth=2, border=2, bg='#FFAA33', highlightbackground="black", highlightthickness=2)
testimg_canvas.place(relx=0.328, rely = 0.285)
label_testimg = tk.Label(window, text= "Test Image", bg='#FFAA33', font=("Comic Sans MS", 10), fg="#ffffff", highlightcolor="black", border=2, highlightthickness=2, borderwidth=2,)
label_testimg.place(relx= 0.33, rely= 0.29)

#Result - title
insimg_canvas = tk.Canvas(window, width=235, height=20, borderwidth=5, border=5, bg="#FFAA33", highlightbackground="black", highlightthickness=4)
insimg_canvas.place(relx=0.075, rely=0.69)
label_result_title = tk.Label(window, text='Result', font=("Papyrus", 11, 'bold'), fg='#000000', bg='#FFAA33')
#label_result_title.configure(font=)
label_result_title.place(relx=0.1, rely=0.696)

#Result - the result
label_result_dir = tk.Label(window, text='None', font=("Arial", 10), fg='#FFAA33', bg='#ffffff')
label_result_dir.place(relx=0.125, rely=0.76)
#black canvas
blank_canvas = tk.Canvas(window, width=235, height=20, borderwidth=5, border=5, bg="#FFAA33", highlightbackground="black", highlightthickness=4)
blank_canvas.place(relx=0.075, rely=0.85)
#Imageless image directory
#insbg = Image.open("./gui/images/imageless.PNG")
insbg = Image.open("src/GUI/images/imageless.PNG")
resize_insbg = insbg.resize((256, 256), Image.Resampling.LANCZOS)
img_inputless = ImageTk.PhotoImage(resize_insbg)

#Image background
label_insbg = tk.Label(window, image=img_inputless, borderwidth=0)
label_insbg.place(relx = 0.33, rely= 0.35)

#Closest Result background - title
cr_canvas = tk.Canvas(window, width=86, height=27, borderwidth=2, border=2, bg='#FFAA33', highlightbackground="black", highlightthickness=2)
cr_canvas.place(relx=0.628, rely = 0.285)
label_testimg = tk.Label(window, text= "Closest Result", bg='#FFAA33', font=("Comic Sans MS", 10), fg="#ffffff")
label_testimg.place(relx= 0.63, rely= 0.295)

#Closest Result background
label_crbg = tk.Label(window, image=img_inputless, bg='#ffffff', borderwidth=0)
label_crbg.place(relx = 0.63, rely= 0.35)

#Execution canvas
execution_canvas = tk.Canvas(width=573, height=65, borderwidth=4, border=4, highlightthickness=4, highlightbackground="#ffffff", bg='#100000')
execution_canvas.place(relx = 0.33, rely = 0.83)
#Execution time - title
label_execution_title = tk.Label(window, text="Execution time: ", font=("Roman", 11, "bold"), fg="#FFAA33", bg="#100000")
label_execution_title.place(relx = 0.335, rely = 0.836)
#Execution time - time
label_execution_time = tk.Label(window, text=(f'00:00'), font=("Courier New", 12), bg='#100000', fg='#5aff15')
label_execution_time.place(relx=0.435, rely= 0.836)

window.mainloop()