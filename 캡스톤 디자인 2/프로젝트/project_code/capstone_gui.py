import tkinter
from tkinter import *
from tkinter import ttk
from math import cos, sin, pi
from tkmacosx import Button
import pygame

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.layers import *
import tensorflow.keras.models as models

from project_code.dicts import *


class Gui(tkinter.Frame):
    def __init__(self, parent):
        self.direction_model = self.load_model('model/final_direction_model.h5')
        self.gender_model = self.load_model('model/final_gender_model.h5')

        self.parent = parent
        self.parent.title("캡스톤디자인 데모")
        self.parent.geometry('640x480')
        self.parent.configure(background='black')

        self.label1 = tkinter.Label(self.parent,
                                    text='테스트할 데이터의 각도를 선택해주세요.',
                                    font=("나눔바른고딕,13"),
                                    fg='white',
                                    bg='black')
        self.label1.place(x=20, y=20)

        combostyle = ttk.Style()
        combostyle.theme_create('combostyle', parent='alt',
                                settings={'TCombobox':
                                              {'configure':
                                                   {'selectbackground': '#F06292',
                                                    'selectforeground': 'white',
                                                    'fieldbackground': 'black',
                                                    'background': 'black',
                                                    'foreground': 'white',
                                                    'relief': 'flat',
                                                    'arrowcolor': '#F06292'
                                                    }}}
                                )
        combostyle.theme_use('combostyle')
        self.angle_str = tkinter.StringVar()
        self.angle_combo = ttk.Combobox(self.parent,
                                        style="myStyle.TCombobox",
                                        width=10,
                                        textvariable=self.angle_str,
                                        font=("나눔바른고딕, 13"),
                                        background='black',
                                        state='readonly')
        self.parent.option_add('*TCombobox*Listbox.font', ("나눔바른고딕, 13"))
        self.parent.option_add('*TCombobox*Listbox.background', 'black')
        self.parent.option_add('*TCombobox*Listbox.foreground', 'white')
        self.parent.option_add('*TCombobox*Listbox.selectBackground', '#F06292')
        self.parent.option_add('*TCombobox*Listbox.selectForeground', 'white')

        self.angle_combo['values'] = ['  0°', '  20°', '  40°', '  60°', '  80°', '  100°', '  120°', '  140°', '  160°', '  180°']
        self.angle_combo.current(0)
        self.angle_combo.place(x=25, y=45)


        self.label2 = tkinter.Label(self.parent,
                                    text='테스트를 진행할 데이터를 선택해주세요.',
                                    font=("나눔바른고딕,13"),
                                    fg='white',
                                    bg='black')
        self.label2.place(x=20, y=80)

        self.data_str = tkinter.StringVar()
        self.data_combo = ttk.Combobox(self.parent,
                                        style="myStyle.TCombobox",
                                        width=10,
                                        textvariable=self.data_str,
                                        font=("나눔바른고딕, 13"),
                                        background='black',
                                        state='readonly')

        self.data_combo['values'] = ['  1번', '  2번', '  3번', '  4번', '  5번']
        self.data_combo.current(0)
        self.data_combo.place(x=25, y=105)

        self.play_bttn = Button(text='  ▶  ',
                                borderless=1,
                                background="#F06292",
                                fg='black',
                                command=lambda: self.play(self.angle_str, self.data_str))
        self.play_bttn.place(x=150, y=103)

        self.predict_bttn = Button(text='  방향 및 정보 예측하기  ',
                                   borderless=1,
                                   background="#F06292",
                                   fg='black',
                                   font=("나눔바른고딕, 13"),
                                   command=lambda: self.predict(self.angle_str, self.data_str))
        self.predict_bttn.place(x=200, y=103)

        self.canvas = tkinter.Canvas(self.parent,
                                     width=600,
                                     height=350,
                                     bg='black',
                                     borderwidth=0,
                                     highlightthickness=0)

        self.photo = PhotoImage(file='/Users/user/git/2015104199/프로젝트/data/gui/drone.gif')
        self.photo = self.photo.subsample(2, 2)
        self.mark = PhotoImage(file='/Users/user/git/2015104199/프로젝트/data/gui/mark.gif')
        self.mark = self.mark.subsample(2, 2)
        self.actual = PhotoImage(file='/Users/user/git/2015104199/프로젝트/data/gui/actual.gif')
        self.actual = self.actual.subsample(2, 2)
        self.mark_label_border = Frame(self.parent, background="#F06292")
        self.actual_label_border = Frame(self.parent, background="grey")
        self.canvas.create_image(300, 320, image=self.photo)
        self.canvas.create_arc(20, 70, 580, 620,
                               start=0,
                               extent=180,
                               fill='',
                               outline='gray',
                               style=ARC,
                               width=3)
        self.canvas.pack(side=BOTTOM)


    def play(self, angle_str, data_str):
        sound = pygame.mixer
        sound.init()
        file_name = pick_data[str(angle_str.get()) + str(data_str.get())]
        s = sound.Sound("/Users/user/git/2015104199/프로젝트/data/test/" + file_name + ".wav")
        s.play()

    def predict(self, angle_str, data_str):
        self.remove()
        self.play(angle_str, data_str)
        file_name = pick_data[str(angle_str.get()) + str(data_str.get())]
        direction_feature = np.load('/Users/user/git/2015104199/프로젝트/data/extract/direction_' + file_name[:-2] + '.npy')[-1]
        gender_feature = np.load('/Users/user/git/2015104199/프로젝트/data/extract/gender_' + file_name[:-2] + '.npy')

        direction_feature = direction_feature[:, :,np.newaxis,]
        direction_predict = self.direction_model.predict(np.array([direction_feature,])).argmax()

        gender_feature = gender_feature[:, :,np.newaxis,]
        gender_predict = self.gender_model.predict(np.array([gender_feature,])).argmax()

        direction_predict_label = direction_labels[direction_predict]
        gender_predict_label = gender_labels[gender_predict]

        x, y = self.cal_location(direction_predict_label)

        self.mark_button = Button(image=self.mark,
                                  bg='black',
                                  relief='flat',
                                  borderless=0,
                                  command=lambda: self.print_label(direction_predict_label, gender_predict_label, file_name))
        self.canvas.create_window(x, y, anchor=CENTER, window=self.mark_button)

        if direction_predict_label != int(file_name.split('_')[1]):
            x, y = self.cal_location(int(file_name.split('_')[1]))

            self.actual_button = Button(image=self.actual,
                                      bg='black',
                                      relief='flat',
                                      borderless=0,
                                      command=lambda: self.print_label(direction_predict_label, gender_predict_label,
                                                                       file_name))
            self.canvas.create_window(x, y, anchor=CENTER, window=self.actual_button)


    def print_label(self, direction_predict_label, gender_predict_label, file_name):
        self.print_predict_info(direction_predict_label, gender_predict_label)
        self.print_actual_info(file_name)

    def print_predict_info(self, direction_predict_label, gender_predict_label):
        if gender_predict_label == 'F': gender = '여자'
        elif gender_predict_label == 'K': gender = '아이'
        else: gender = '남자'

        kwargs = {
            "direction": str(direction_predict_label),
            "gender": gender
        }

        info = """
  목소리의 발원 방향 : {direction}°  
  발화자 정보 : {gender}  
        """.format(**kwargs)
        self.mark_info_label = tkinter.Label(self.mark_label_border,
                                        text=info,
                                        font=("나눔바른고딕,13"),
                                        fg='white',
                                        bg='black',
                                        bd=0)
        self.mark_info_label.pack(fill="both", expand=True, padx=3, pady=3)
        self.mark_label_border.place(x=130, y=320)

    def print_actual_info(self, file_name):
        actual_info = file_name.split('_')
        direction_actual_label = actual_info[1]
        gender_actual_label = actual_info[-1]

        if gender_actual_label == 'F':
            gender = '여자'
        elif gender_actual_label == 'K':
            gender = '아이'
        else:
            gender = '남자'

        kwargs = {
            "direction": str(direction_actual_label),
            "gender": gender
        }

        info = """
    목소리의 발원 방향 : {direction}°  
    발화자 정보 : {gender}  
          """.format(**kwargs)

        self.actual_info_label = tkinter.Label(self.actual_label_border,
                                        text=info,
                                        font=("나눔바른고딕,13"),
                                        fg='white',
                                        bg='black',
                                        bd=0)
        self.actual_info_label.pack(fill="both", expand=True, padx=3, pady=3)
        self.actual_label_border.place(x=320, y=320)

    def remove(self):
        self.canvas.delete("all")
        self.actual_label_border.destroy()
        self.mark_label_border.destroy()

        self.mark_label_border = Frame(self.parent, background="#F06292")
        self.actual_label_border = Frame(self.parent, background="grey")
        self.canvas.create_image(300, 320, image=self.photo)
        self.canvas.create_arc(20, 70, 580, 620,
                               start=0,
                               extent=180,
                               fill='',
                               outline='gray',
                               style=ARC,
                               width=3)
        self.canvas.pack(side=BOTTOM)

    def start(self):
        self.parent.mainloop()

    def cal_location(self, direction_label):
        x = 300 - 280 * cos(radian_data[direction_label] / 9 * pi)
        y = 350 - 280 * sin(radian_data[direction_label] / 9 * pi)
        return x, y



    def load_model(self, model_path):
        loaded_model = models.load_model(PROJECT_DIR+model_path, custom_objects={'LeakyReLU': LeakyReLU()})
        return loaded_model

root = tkinter.Tk()
main = Gui(root)
main.start()