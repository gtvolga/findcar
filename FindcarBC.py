import os
import cv2
import numpy as np
from math import floor
from keras.models import load_model
from keras.optimizers import Adam
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
from tkinter import filedialog
from PIL import Image, ImageTk
import time

def open_file():
    open_file.filename = filedialog.askopenfilename(initialdir = "F:/Учеба/Проект/6сем/Data",
                                          title = "Select a File",
                                          filetypes = (("Image files",
                                                        "*.jpg*"),
                                                       ("all files",
                                                        "*.*")))
    dl.configure(text="Открыт файл: "+os.path.basename(open_file.filename))
    if open_file.filename is not None:
        pass

def open_mod():
    open_mod.modname = filedialog.askopenfilename(initialdir = "F:/Учеба/Проект/6сем",
                                          title = "Select a File",
                                          filetypes = (("Net files",
                                                        "*.h5*"),
                                                       ("all files",
                                                        "*.*")))
    ml.configure(text="Открыт файл: "+os.path.basename(open_mod.modname))
    if open_mod.modname is not None:
        pass

def detect_car():
    resl = Label(ws)
    model = load_model(open_mod.modname)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(1e-5),
                  metrics=['accuracy'])    
    img = cv2.imread(open_file.filename)
    img = cv2.resize(img, (150, 150))
    img = np.reshape(img, [1, 150, 150, 3])
    img = img / 255
    pr = truncate(model.predict(img)[0][0], 2)
    if (pr > 0.6):
        resl = Label(ws, text=f'На фото автомобиль класса C, {pr*100}%', foreground='blue', font=("Times New Roman", 18)).grid(row=4, columnspan=3, pady=10)
    else:
        resl = Label(ws, text=f'На фото автомобиль класса B, {100 - pr*100}%', foreground='green', font=("Times New Roman", 17)).grid(row=4, columnspan=3, pady=10)
    image = Image.open(open_file.filename).resize((300, 400))
    photo = ImageTk.PhotoImage(image)
    imgl = Label(image = photo)
    imgl.image = photo
    imgl.grid(row = 4, column = 5, padx = 5, pady = 5)

def truncate(n, dec = 0):
    mult = 10 ** dec
    res = int(n * mult) / mult
    if res == 0.00:
        res = 0.01
    return res

if __name__ == "__main__":
    ws = Tk()
    ws.title('Распознавание автомобиля')
    ws.geometry('675x525')
    dl = Label(ws, text='Выбрать фотографию')
    dl.grid(row=1, column=0, padx=10)
    dlbtn = Button(ws, text ='Выбрать', command = lambda:open_file())
    dlbtn.grid(row=1, column=1)

    ml = Label(ws, text='Выбрать модель')
    ml.grid(row=2, column=0, padx=10)
    mlbtn = Button(ws, text ='Выбрать', command = lambda:open_mod()) 
    mlbtn.grid(row=2, column=1)

    upld = Button(ws, text='Определить', command = lambda:detect_car())
    upld.grid(row=3, columnspan=3, pady=10)
    ws.mainloop()