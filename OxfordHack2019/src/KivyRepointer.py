import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.graphics import *
from kivy.uix.widget import Widget
​
class MyWidget(Widget):
    balls=[]
    
​
    def __init__(self, **kwargs):
        super(MyWidget, self).__init__(**kwargs)
        self.radius = 80
​
       
        Color(1, 1, 0)
        with self.canvas:
            self.rect = Rectangle(pos=self.pos, size=self.size)
​
        self.bind(pos=self.update_rect)
        self.bind(size=self.update_rect)
        with self.canvas:
            Color(1, 0.5, 0)
            rec = Rectangle(pos=(500,500), size=(121*4,91*4))
    def addBalls(self,manyballs):
        self.balls.append( manyballs)
        
    def drawABall(self):
        with self.canvas:
            Color(1, 0.5, 0)
            #d=ball.radi*2
            Ellipse(pos=(70,70), size=(self.radius, self.radius))
        
    def drawBall(self):
        for ball in self.balls: 
            drawABall(ball)
               
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
        
    def on_touch_down(self,touch):
        with self.canvas:
            
            d = 30.
            
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            self.drawABall()
​
class Ball():
    def __init__(self, **kwargs):
        super(Ball, self).__init__(**kwargs)
        self.radius = 4*5.7
    def colorize(self,color):
        self.color = color
    def position(self,position):
        self.pos = position
    def display(self,a_widget):
        with a_widget.canvas:
            Ellipse(pos=self.pos, size=(self.radius, self.radius))
            
            
class MyApp(App):
    
       
    def build(self):
        self.widget = MyWidget()
        return self.widget
​
    def drawBalls(self):
        print("lol")
        self.widget.drawABall()
​
    def addBalls(self, balls):
        self.widget.addBalls(balls)
        print(len(self.widget.balls))
        

        
​
if __name__=='__main__':
    app = MyApp()
    app.balls = 3
    app.build() 
​
    ball = Ball()
    ball.colorize(Color(1,1,0))
    ball.position((20,30))
    ball.display(app.widget)
    
    
    app.addBalls(ball)
    app.drawBalls()
        
    
    app.run()