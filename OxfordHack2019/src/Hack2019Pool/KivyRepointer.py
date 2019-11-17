from kivy.app import App

from kivy.uix.boxlayout import *
from kivy.uix.gridlayout import *
from kivy.uix.label import *
from kivy.uix.textinput import *
from kivy.uix.button import *
from kivy.uix.camera import *
from kivy.uix.image import *
from kivy.uix.dropdown import *

import time

class MainScreen(GridLayout):

    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.cols = 1

        PLAYERS = ["Undecided","Red","Yellow"]
        self.player = "Undecided"
        
        self.top_row = BoxLayout(orientation = "horizontal", size_hint_y = 0.8)
        self.bottom_row = BoxLayout(orientation = "horizontal", size_hint_y = 0.2)
        self.add_widget(self.top_row)
        self.add_widget(self.bottom_row)
        self.top_widgets = []
        self.bottom_widgets = []

        self.camera = Camera(play = True, resolution = (960,640), size_hint_x = 1.5)
        self.top_row.add_widget(self.camera)
        self.top_widgets.append(self.camera)

        self.image = Image(source = "table.png") #placeholder picture
        self.top_row.add_widget(self.image)
        self.top_widgets.append(self.image)

        self.button_camera = Button(text='Take a pic!', font_size=30)
        self.button_balls = Button(text='Add yellow balls', font_size=20)

        def capture(self):
            '''
            Function to capture the images and give them the names
            according to their captured time and date.
            '''
            timestr = time.strftime("%Y%m%d_%H%M%S")
            self.camera.export_to_png("IMG_{}.png".format(timestr))
            print("Captured")


        def take_pic(instance):
            if self.camera.play:
                capture(self)
                self.button_camera.text = "Retake?"
            else:
                self.button_camera.text = "Take a pic!"
            self.camera.play = not(self.camera.play)
        
        def swap_balls(instance):
            if self.button_balls.text == 'Add yellow balls':
                self.button_balls.text = 'Add red balls'
            elif self.button_balls.text == 'Add red balls':
                self.button_balls.text = 'Add white ball'   
            elif self.button_balls.text == 'Add white ball':
                self.button_balls.text = 'Add black ball'
            else:
                self.button_balls.text = 'Add yellow balls'

        self.button_camera.bind(on_press = take_pic)
        self.button_balls.bind(on_press = swap_balls)
        self.bottom_row.add_widget(self.button_camera)
        self.bottom_widgets.append(self.button_camera)
        self.bottom_row.add_widget(self.button_balls)
        self.bottom_widgets.append(self.button_balls)

        self.button_calculate = Button(text='Calculate!')
        self.bottom_row.add_widget(self.button_calculate)
        self.bottom_widgets.append(self.button_calculate)

        def null_function(instance):
            return None
        
        

        def calculate(instance):
            self.button_calculate.bind(on_press = null_function)
            self.button_calculate.text = "Loading..."
            self.button_calculate.bind(on_press = calculate)
            self.button_calculate.text = "Calculate!"
        
        self.button_calculate.bind(on_press = calculate)




        self.button_player = Button(text="Player: " + self.player)
        self.bottom_row.add_widget(self.button_player)
        self.bottom_widgets.append(self.button_player)
        
        def change_player(instance):
            self.player = PLAYERS[(PLAYERS.index(self.player)+1)%3]
            self.button_player.text = "Player: " + self.player
        self.button_player.bind(on_press = change_player)


class MyApp(App):

    def build(self):
        return MainScreen()


if __name__ == '__main__':
    MyApp().run()
