from kivy.app import App

from kivy.config import Config
Config.set("graphics", "resizable", False)

from kivy.uix.boxlayout import *
from kivy.uix.gridlayout import *
from kivy.uix.label import *
from kivy.uix.textinput import *
from kivy.uix.button import *
from kivy.uix.widget import Widget
from kivy.uix.camera import *
from kivy.graphics import *
from kivy.graphics.instructions import *
from kivy.uix.image import *
from kivy.uix.dropdown import *
from kivy.uix.floatlayout import *

from kivy.clock import Clock
from kivy.core.window import Window



import time


import random
import cv2 as cv
import numpy as np
from cv2 import line

import matplotlib
import matplotlib.pyplot as plt

class ImageProcessor(object):


    def __init__(self, url):
        
        self.baseUrl = "C:\\Users\\George\\Pictures\\Hack_tests"
        
        self.initial = cv.imread(url)
        #cv.imshow('initial', self.initial)
    
    def label_balls(self, im, tableWidth):
        imcopy = im.copy()
        
        circles, whiteBall, blackBall = self.extract_circles(im, tableWidth)
        valids = []
        
        #self.show_image("label", im)
        #cv.imwrite("C:\\Users\\George\\Pictures\\Hack_tests\circle_stuff.jpg", im)
        
        for i,circle in enumerate(circles[0,:]):
            x,y = (int(circle[0]),int(circle[1]))
            rad = int(circle[2])

            if 0<x-rad//2 and x+rad//2<im.shape[1] and y-rad//2>0 and y+rad//2< im.shape[0]:
                
                working = True
                
                
                for x1 in range(x-rad//2, x+rad//2, rad//10):
                    for y1 in range(y-rad//2,y+rad//2, rad//10):
                        (b,g,r) = im[y1,x1]
                        if (b,g,r) == (0,0,0) or g > max(r,b)*1.2:
                            working = False
                if working:
                    valids.append([x,y,int(circle[2])])       
        
        
        hList = []
        for i,circle in enumerate(valids):
            x,y = circle[0], circle[1]
               
            #print(circle)
            hSum = 0
            hCount = 0
            if 0<x-rad//2 and x+rad//2<im.shape[1] and y-rad//2>0 and y+rad//2< im.shape[0]:   
                for x1 in range(x-rad//2, x+rad//2, rad//10):
                    for y1 in range(y-rad//2,y+rad//2, rad//10):
                        hCount += 1
                        pix = np.uint8([[imcopy[y,x]]])
                        #print(pix)
                        hsv = cv.cvtColor(pix, cv.COLOR_BGR2HSV)
                        h,s,v = cv.split(hsv)
                        hSum += h[0][0]
                
                hList.append(hSum/hCount)
        
        hAverage = sum(hList) / len(hList)
        print(hAverage)
        
        for i,circle in enumerate(valids):
            
            if hAverage < 50:
                if hList[i] > hAverage:
                    circle.append('R')
                else:
                    circle.append('Y')
            else:
                if hList[i] < hAverage:
                    circle.append('R')
                else:
                    circle.append('Y') 
                
        whiteBall.append("W")
        blackBall.append("B")
        allBalls = valids + [whiteBall] + [blackBall]
        print(allBalls)
        
        try:
            for i in allBalls:
                if i[3] == 'Y':
                    col = (255,255,0)
                if i[3] == 'R':
                    col = (0,255,255)
                if i[3] == 'W':
                    col = (0,0,0)
                if i[3] == 'B':
                    col = (255,255,255)
                cv.circle(im,(i[0],i[1]),i[2],col,2)
                cv.circle(im,(i[0],i[1]),2,col,3)
        except:
            pass
        #self.show_image("circle image2", im)
        return allBalls, im
        
    def get_white_ball(self, im, tableWidth,expected):        
        whiteImage = im.copy()
        whiteImage = self.filter_for_white(whiteImage)
        whiteImage = cv.cvtColor(whiteImage, cv.COLOR_BGR2GRAY)
        whiteCircle = cv.HoughCircles(whiteImage,cv.HOUGH_GRADIENT,10,200,
                            param1=60,param2=30,
                            minRadius=int(expected*0.2),maxRadius=int(expected*1.5))
        
        whiteBall = None
        
        valids = []
        for i,circle in enumerate(whiteCircle[0,:10]):
            x,y = (int(circle[0]),int(circle[1]))
            rad = int(circle[2])

            if 0<x-rad//2 and x+rad//2<im.shape[1] and y-rad//2>0 and y+rad//2< im.shape[0]:
                
                working = 0
                
                
                for x1 in range(x-rad//2, x+rad//2, rad//10):
                    for y1 in range(y-rad//2,y+rad//2, rad//10):
                        (b,g,r) = im[y1,x1]
                        if (b,g,r) == (0,0,0) or g > max(r,b)*1.2 or int(r)*int(g)*int(b) < 150*150*150:
                            working += 1
                if working == 0 and whiteBall == None:
                    whiteBall = [x,y,int(circle[2])]
                    print("---------------------", whiteBall)
                else:
                    valids.append([working,x,y,int(circle[2])])
        
        if whiteBall == None:
            v = valids[0]
            v.pop(0)
            whiteBall = v
        

        return whiteBall

    def get_black_ball(self, im, tableWidth,expected):        
        blackImage = im.copy()
        blackImage = self.filter_for_black(blackImage)
        blackImage = cv.cvtColor(blackImage, cv.COLOR_BGR2GRAY)
        #self.show_image("black image for circles", blackImage)
        blackCircle = cv.HoughCircles(blackImage,cv.HOUGH_GRADIENT,10,200,
                            param1=60,param2=30,
                            minRadius=int(expected*0.2),maxRadius=int(expected*1.5))
        
        blackBall = None
        
        valids = []
        for i,circle in enumerate(blackCircle[0,:10]):
            x,y = (int(circle[0]),int(circle[1]))
            rad = int(circle[2])

            if 0<x-rad//2 and x+rad//2<im.shape[1] and y-rad//2>0 and y+rad//2< im.shape[0]:
                
                working = 0
                
                
                for x1 in range(x-rad//2, x+rad//2, rad//10):
                    for y1 in range(y-rad//2,y+rad//2, rad//10):
                        (b,g,r) = im[y1,x1]
                        if (b,g,r) == (0,0,0) or g > max(r,b)*1.2 or int(r)*int(g)*int(b) < 150*150*150:
                            working += 1
                if working == 0 and blackBall == None:
                    blackBall = [x,y,int(circle[2])]
                    print("---------------------", blackBall)
                else:
                    valids.append([working,x,y,int(circle[2])])
        
        if blackBall == None:
            v = valids[0]
            v.pop(0)
            blackBall = v
        
        print('black ball', blackBall)
        return blackBall
    
    
        
    def extract_circles(self, im, tableWidth):
        
        #expected = 0.05 * tableWidth
        expected = 75
        
        whiteBall = self.get_white_ball(im, tableWidth, expected)
        
        blackBall = self.get_black_ball(im, tableWidth, expected)
        
        circleImage = im.copy()
        
        copy = im.copy()
        #im = cv.circle(copy, (whiteBall[0], whiteBall[1]), whiteBall[2], (0,0,0), -1)
        
        im = self.filter_for_balls(im)
        #self.show_image("filtered", im)
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        #im = cv.Canny(im, 100, 200)
        
        #im = cv.bitwise_not(im)
        #kernel = np.ones((10,10),np.float32)/4
        #im = cv.filter2D(im,-1,kernel)
        #self.show_image("blur", blurred)
        #self.show_image("cannied", im)
        circles = cv.HoughCircles(im,cv.HOUGH_GRADIENT,10,100,
                            param1=60,param2=30,
                            minRadius=int(expected*0.2),maxRadius=int(expected*1.5))
        try:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv.circle(circleImage,(i[0],i[1]),i[2],(255,255,0),2)
                cv.circle(circleImage,(i[0],i[1]),2,(255,255,0),3)
        except:
            pass
        
        #self.show_image("circle image", circleImage)
        return circles, whiteBall, blackBall
    
    def filter_for_white(self,im):
        #=======================================================================
        # average = im.mean(axis=0).mean(axis=0)
        # print(average)
        #=======================================================================
        lowerFilter = np.array([175, 175,175])
        upperFilter = np.array([255,255,255])
        
        mask = cv.inRange(im, lowerFilter, upperFilter)
        newIm = cv.bitwise_and(im,im, mask = mask)
        
        
        #self.show_image("white image", newIm)

        return newIm

    def filter_for_black(self,im):
        #=======================================================================
        # average = im.mean(axis=0).mean(axis=0)
        # print(average)
        #=======================================================================
        im = cv.bitwise_not(im)
        
        #self.show_image("negative image", im)
        
        lowerFilter = np.array([200, 200,200])
        upperFilter = np.array([255,255,255])
        
        mask = cv.inRange(im, lowerFilter, upperFilter)
        newIm = cv.bitwise_and(im,im, mask = mask)
        
        #self.show_image("black image", newIm)

        return newIm
    
    def filter_for_balls(self, im):
        
        hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)

        lowerYellow = np.array([0,0,0])
        upperYellow = np.array([35,255,255])
    
        mask = cv.inRange(hsv, lowerYellow, upperYellow)
        res = cv.bitwise_and(im,im, mask= mask)
        
        newIm = cv.cvtColor(res, cv.COLOR_HSV2BGR)
        
        newIm = cv.bitwise_not(newIm)

        lowerFilter = np.array([255,255,255])
        upperFilter = np.array([255,255,255])
        
        mask = cv.inRange(newIm, lowerFilter, upperFilter)
        newIm = cv.bitwise_and(newIm,newIm, mask = mask)        
        
        newIm = cv.bitwise_not(newIm)
        return newIm
    
    def cut_board(self, lines):
        im = self.initial.copy()
        (x1,y1), (x2,y2), (x3,y3), (x4,y4) = self.get_corners(lines)
        
        pts = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
        
        #make bounding rectangle and crop to it
        rect = cv.boundingRect(pts)
        x,y,w,h = rect
        croped = im[y:y+h, x:x+w].copy()
        
        #make mask
        pts = pts - pts.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
 
        # apply mask
        dst = cv.bitwise_and(croped, croped, mask=mask)
        
        #print(lines)
        angle = np.arctan(lines[3][0][0]) * (180/np.pi)
        #print(angle)
        #self.show_image('not rotated', dst)
        rows,cols = dst.shape[0], dst.shape[1]

        M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
        dst2 = cv.warpAffine(dst,M,(cols,rows))
        
        #self.show_image('rotated', dst2)

        return dst2
        
    def get_corners(self, lines):
        coords = []
        ijs = [(0,2), (0,3), (1,3), (1,2)]
        for (i,j) in ijs:  
            (m1,c1), (m2,c2) = lines[i][0], lines[j][0]
            x = (c2-c1) / (m1-m2)
            y = m1*x + c1
            coords.append((int(x),int(y)))
        return coords
    
    def extract_board(self):
        greenIm = self.filter_green(self.initial)
        
        #smallGreenIm = self.resize(greenIm, 0.2)
        #self.show_image("filtered_green", smallGreenIm)
        
        whiteFreeIm = self.remove_white(greenIm)
        #self.show_image('white filtered', whiteFreeIm)
        
        refilteredIm = self.exclude_rb(whiteFreeIm)
        #self.show_image('refiltered', refilteredIm)
        #cv.imwrite(self.baseUrl + "\refiltered.png", refilteredIm)
        
        edgesIm = self.do_edge_detection(refilteredIm)
        #self.show_image('edges2', edgesIm)
        
        lineList = self.get_lines(edgesIm)
        #self.display_lines(self.initial, lineList)
        #self.display_xy_lines(self.initial, lineList)
        
        #print(lineList)
        return lineList   
        
    def show_image(self,label, im):
        #print(im.shape)
        width, height = im.shape[0], im.shape[1]
        #print(width, height)
        if width > 800:
            factor = 800/width
            #print(factor)
            im = self.resize(im, factor)
        
        cv.imshow(label, im)   
    
    def filter_green(self, im):
        (b,g,r) = cv.split(im)
        
        m = np.maximum(np.maximum(b,g), r)

        g[g<m] = 0
        
        newIm = cv.merge([b,g,r])
        return newIm
    
    def remove_white(self, im):
                
        lowerFilter = np.array([0, 0,0])
        upperFilter = np.array([200,255,200])
        
        mask = cv.inRange(im, lowerFilter, upperFilter)
        newIm = cv.bitwise_and(im,im, mask = mask)

        return newIm

    def exclude_rb(self, im):
        # set green value for pixels that have too much red and blue to 0
        # then cut out pixels with 0 green value
        
        (b,g,r) = cv.split(im)
        
        m = np.minimum(np.minimum(b,g), r)

        g[g<(1.5*m)] = 0
        
        newIm = cv.merge([b,g,r])
        
        # the one here is important
        lowerFilter = np.array([0, 1,0])
        upperFilter = np.array([200,255,200])
        
        mask = cv.inRange(newIm, lowerFilter, upperFilter)
        newIm = cv.bitwise_and(newIm,newIm, mask = mask)

        return newIm
    
    def do_edge_detection(self,im):
        newImage = cv.Canny(im, 100, 200)
        return newImage
    
    def get_lines(self, edgeImage):
        #                     target,   rho, theta, thresh required to identify
        lines = cv.HoughLines(edgeImage,1,np.pi/180,200)
        hLines = []
        vLines = []
        
        for line in lines:
            for (rho, theta) in line:
                lineLength = 10000
                cT = np.cos(theta)
                sT = np.sin(theta)
                x0 = cT*rho
                y0 = sT*rho
                x1 = int(x0 + lineLength*(-sT))
                y1 = int(y0 + lineLength*(cT))
                x2 = int(x0 - lineLength*(-sT))
                y2 = int(y0 - lineLength*(cT))
                m = (y2-y1) / (x2-x1)
                c = y2 - m*x2
                if 1 < theta < 2:
                    hLines.append([(m, c)])
                else:
                    vLines.append([(m, c)])
                    
        #self.display_lines(self.initial, hLines)
        halfY = edgeImage.shape[1] // 2
        closest = None
        closestX = 10000
        furthest = None
        furthestX = 0
        for line in vLines:
            m,c = line[0]
            x = (halfY - c)/m
            if x < closestX:
                closestX = x
                closest = line
            if x > furthestX:
                furthestX = x
                furthest = line
        
        newLines = [closest, furthest]
        
        halfX = edgeImage.shape[0] // 2
        closest = None
        closestY = 10000
        furthest = None
        furthestY = 0
        for line in hLines:
            m,c = line[0]
            y = m*halfX + c
            if y < closestY:
                closestY = y
                closest = line
            if y > furthestY:
                furthestY = y
                furthest = line
            
        newLines = newLines + [closest, furthest]
        
        return newLines
        
    
    def display_lines(self, im, lineList):
        imCopy = im.copy()
        for line in lineList:
            for (rho,theta) in line:
                lineLength = 10000
                cT = np.cos(theta)
                sT = np.sin(theta)
                x0 = cT*rho
                y0 = sT*rho
                x1 = int(x0 + lineLength*(-sT))
                y1 = int(y0 + lineLength*(cT))
                x2 = int(x0 - lineLength*(-sT))
                y2 = int(y0 - lineLength*(cT))
            
            cv.line(imCopy,(x1,y1),(x2,y2),(0,0,255),5)
    
        #self.show_image("line image" + str(random.randint(0,1000)), imCopy)

    def display_xy_lines(self, im, lineList):
        imCopy = im.copy()
        for line in lineList:
            for (m,c) in line:
                x1 = -10000
                y1 = int(m*x1 + c)
                x2 = 10000
                y2 = int(m*x2 + c)
            
            cv.line(imCopy,(x1,y1),(x2,y2),(0,0,255),5)
    
        #self.show_image("line image" + str(random.randint(0,1000)), imCopy)
    

    def resize(self, im, factor):
        return cv.resize(im, (0,0), fx=factor, fy=factor)

class Projector(object):

    def __init__(self, balls, image, url):
        #print(lines)
        
        #print(angle)
        
        i = ImageProcessor(url)
        i.initial = image
        lines = i.extract_board()
        angle = np.arctan(lines[3][0][0])
        
        i.display_xy_lines(image, lines)
    
        newList = []
        
        for ball in balls:
            x,y,rad,type = ball
            x1 = (y-lines[0][0][1])/lines[0][0][0]
            x2 = (y-lines[1][0][1])/lines[1][0][0]
            dist = abs (x1-x2)
            xratio = (x-x1) / dist
            
            yratio = 1 - (y / image.shape[1]) ** 0.5
            
            #self.project(image, x, y, lines)
            
            print('x',xratio)
            print('y',yratio)
        
            newList.append([xratio, yratio*2, type])

        self.newList = newList
         
        #=======================================================================
        # fig, ax = plt.subplots(figsize=(12,6))
        #  
        # plt.plot(yList, xList, marker='x', color='black', linestyle='None', markersize = 5.0)
        #  
        # axes = plt.gca()
        # axes.set_xlim([0,1])
        # axes.set_ylim([0,2])
        #  
        #  
        #  
        # plt.show()
        #=======================================================================
    
    def project(self, image, x, y, lines):
        width, height = image.shape[0], image.shape[1]
        A = (width//2, height)
        B = (x,y)
        C = self.get_halfway(image, lines)
        D = (width//2, 0)
        
        ac = 1
        BC = self.get_dist(B,C)
        AD = self.get_dist(A,D)
        AC = self.get_dist(A,C)
        ad = 2
        cd = 1
        BD = self.get_dist(B,D)
        
        bc = cd / ((ad*AC*BD / (ac*BC*AD)) - 1)

        dis = 1- bc

        print('bc is  ', bc)

        return dis
    
    def get_dist(self, coord1, coord2):
        return abs(coord1[1]-coord2[1])
    
    def get_halfway(self,image, lines):
        height,width = image.shape[0], image.shape[1]
        #cv.imshow("im", cv.resize(image.copy(), (0,0), fx=0.2, fy=0.2))
        
        m,c = lines[1][0]
        x1 = -10000
        y1 = int(m*x1 + c)
        x2 = 10000
        y2 = int(m*x2 + c)
        
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),5)
        
        holex,holey = [], []
        
        for y in range(height//6, height//3, 1):
            x = int((y-c)/ m) - 5
            if x > 0 and x < width:
                b,g,r = image[y,x]
                if b<30 and g < 30 and r <30:
                    holex.append(x)
                    holey.append(y)
            
        if holex != [] and holey != []:
            hole1 = [sum(holex) / len(holex), sum(holey)/ len(holey)]
        else:
            hole1 = None
            raise Exception ("Ahhhh")
        
        
        m,c = lines[0][0]
        holex,holey = [], []
        
        for y in range(height//6, height//3, 1):
            x = int((y-c)/ m) + 5
            if x > 0 and x < width:
                b,g,r = image[y,x]
                if b<30 and g < 30 and r <30:
                    holex.append(x)
                    holey.append(y)
            
        if holex != [] and holey != []:
            hole2 = [sum(holex) / len(holex), sum(holey)/ len(holey)]
        else:
            hole2 = None
            raise Exception ("Ahhhh")
        
        halfWayMark = [width//2, int(hole1[1] + hole2[1]+50)//2]
        cv.circle(image, (halfWayMark[0],halfWayMark[1]), 10, (0,255,255), 3 )
        #cv.imshow("im2", cv.resize(image.copy(), (0,0), fx=0.2, fy=0.2))
        return halfWayMark
        
    def rotate_coords(self, coords, origin, radians):
        x, y = coords
        ox, oy = origin
    
        qx = ox + np.cos(radians) * (x - ox) + np.sin(radians) * (y - oy)
        qy = oy + -np.sin(radians) * (x - ox) + np.cos(radians) * (y - oy)
    
        return qx, qy



#----------------------------------------------------------------------------------------------------

def transform_coordinates(pos):
        return (2.2*pos[0]+12, 3.5*pos[1] -180)
    
def inverse_transform(pos):
    return ((pos[0]-12)/2.2,(pos[1]+180)/3.5)

class PoolTable(Widget):
    
    def __init__(self):
        super(PoolTable, self).__init__()
        
        Window.size = (800,600)
        
        self.radius = 80
        self.balls = []
        self.drawn_balls = []
        self.active_height = Window.size[1]*0.8
        self.active_width = self.active_height/960*562
        self.active_size = (self.active_width,self.active_height)
        print(self.active_size)
        self.lower_pos = (Window.size[0]-self.active_width,Window.size[1]-self.active_height)
        
        with self.canvas:
            Rectangle(source= "C:\\Users\\George\\Pictures\\Hack_tests\\pool_table.png",pos=self.lower_pos, size=(self.active_width,self.active_height))
        
    def drawBall(self, ball):
        with self.canvas:
            ball.color
            Ellipse(pos=(self.lower_pos[0]+ball.pos[0]+ball.radius,self.lower_pos[1]+ball.pos[1]+ball.radius), size=(ball.radius, ball.radius))
      
    def drawBalls(self):
        for ball in self.balls:
            ball.display(self)
                       
    def addBalls(self,manyballs):
        self.canvas.clear()
        with self.canvas:
            Rectangle(source= "C:\\Users\\George\\Pictures\\Hack_tests\\pool_table.png",pos=self.lower_pos, size=(self.active_width,self.active_height))

 
            self.balls = manyballs
            self.drawBalls()
        
class Ball():
    def __init__(self, color,position):
        super(Ball, self).__init__()
        self.radius = 2*5.7
        self.color = color
        self.pos = transform_coordinates(position)
        print(self.pos)
        
    def display(self,a_widget):
        print(self.color)
        with a_widget.canvas:
            Color(self.color[0], self.color[1], self.color[2])
            Ellipse(pos=(a_widget.lower_pos[0]+self.pos[0]+self.radius,a_widget.lower_pos[1]+self.pos[1]+self.radius), size=(self.radius, self.radius))
            
          
#===============================================================================
# class MyApp(App):
#     
#     def build(self):
#         self.widget = PoolTable(width = Window.size[0]/4, height = Window.size[1]*4/5)
#         return self.widget
#     
#     def drawBalls(self):
#         print("lol")
#         self.widget.drawBall()
#         
#     def addBalls(self, balls):
#         self.widget.addBalls(balls)
#         print(len(self.widget.balls))
#===============================================================================
      
      
#===============================================================================
# if __name__=='__main__':
#     app = MyApp()
#     app.balls = 3
#     app.build() 
# ​
#     ball = Ball()
#     ball.colorize(Color(1,1,0))
#     ball.position((20,30))
#     #ball.display(app.widget)
#     
#     
#     app.addBalls(ball)
#     app.drawBalls()
#         
#     
#     app.run()
#===============================================================================


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

        #self.camera = Camera(play = True, resolution = (960,640), size_hint_x = 1.5)
        
        url = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191115_191414.jpg" #  "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191116_100356.jpg"
        self.camera = Image(source = url)
        self.top_row.add_widget(self.camera)
        self.top_widgets.append(self.camera)

        self.table = PoolTable() #placeholder picture
        
        self.top_row.add_widget(self.table)
        self.top_widgets.append(self.table)

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
            if self.button_camera.text == "Take a pic!":
                timestr = time.strftime("%Y%m%d_%H%M%S")
                url = "IMG_{}.png".format(timestr)
                
                #url = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191116_100356.jpg"
                url = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191115_191414.jpg"
                print('changed')
                self.button_camera.text = "Loading..."
                
                try:
                    i = ImageProcessor(url)
                    
                    lines = i.extract_board()
                    cutBoard = i.cut_board(lines)
                    
                    balls,ballImage = i.label_balls(cutBoard, 2500)
                    cv.imwrite("C:\\Users\\George\\Pictures\\Hack_tests\\processed.jpg", ballImage)
                    #cv.imshow("ball image", ballImage)
                    self.camera = Image(source = "C:\\Users\\George\\Pictures\\Hack_tests\\processed.jpg")
                                        
                    proj = Projector(balls, cutBoard, url)
                    self.ballList = proj.newList
                    
                    print(self.ballList)
                    
                    for i in range(len(self.ballList)):
                        x = int(self.ballList[i][0] * 100)
                        y = int(self.ballList[i][1] * 100)
                        col = self.ballList[i][2]
                        if col == 'R':
                            col = [1,1,0]
                        elif col == "Y":
                            col = [1,0,0]
                        elif col == "B":
                            col = [0,0,0]
                        else:
                            col = [1,1,1]
                        
                        self.ballList[i][2] = col
                    
                        self.ballList[i] = Ball(col, (x,y))
                    
                    self.table.addBalls(self.ballList)
                               
                    
                except Exception as e:
                    print(e)
                    self.button_camera.text = "Try again"
                
                if self.button_camera.text in ["Take a pic!", "Loading..."]:
                    self.button_camera.text = "Retake?"
                
            else:
                self.button_camera.text = "Take a pic!"
                self.camera.source = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191115_191414.jpg"
            
            self.camera.reload()
            #self.camera.play = not(self.camera.play)
        
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
            # give self.ballList to foo()
            
            
            for i, ball in enumerate(self.ballList):
                col = ball.color
                if col == [1,1,0]:
                    col = 'red'
                elif col == [1,0,0]:
                    col = 'yellow'
                elif col == [0,0,0]:
                    col = 'black'
                else:
                    col = 'white'
                
                ballPos = inverse_transform(ball.pos)
                self.ballList[i] = [[ballPos[0], ballPos[1]], col]
            
            if self.player == 'Undecided':
                colours = ['red', 'yellow']
            else:
                colours = [self.player.lower()]
            
            dict = turn(colours, self.ballList)
            
            print(dict)
            
            cuePoint = dict['cue position']
            whitePoint = dict['collision point']
            pocketPoint = dict['target hole']
            
            print(cuePoint, whitePoint, pocketPoint)
            cuePoint = [cuePoint[0], cuePoint[1]]
            whitePoint = [whitePoint[0], whitePoint[1]]
            pocketPoint = [pocketPoint[0], pocketPoint[1]]
            
            cuePoint = transform_coordinates(cuePoint)
            whitePoint = transform_coordinates(whitePoint)
            pocketPoint = transform_coordinates(pocketPoint)
            
            with self.canvas:
                p1 = [cuePoint[0]+self.table.lower_pos[0]+4*5.7, cuePoint[1]+self.table.lower_pos[1]+4*5.7, whitePoint[0]+self.table.lower_pos[0]+4*5.7, 4*5.7+ whitePoint[1]+self.table.lower_pos[1]]
                print(p1)
                Line(points=p1)
                Line(points=[whitePoint[0]+self.table.lower_pos[0]+4*5.7, whitePoint[1]+self.table.lower_pos[1]+4*5.7, pocketPoint[0]+4*5.7+self.table.lower_pos[0], pocketPoint[1]+self.table.lower_pos[1]+4*5.7])
            
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

 
 
#-------------------------------------------------------------------------------


#Input = position of every ball and dimension of table
#Output = optimal shot coordinates





import math




info = {
    "resolution" : 1,
    "dimensions" : [100,200],
    "corner_len" : 8,
    "side_len" : 6,
}



#useful functions for geometry

def euclidean_dist(pos1,pos2):
    return ((pos1[1]-pos2[1])**2 + (pos1[0]-pos2[0])**2)

def slope(pos1,pos2):
    if pos1[0]==pos2[0]:
        return 1000
    else:
        
        return (pos1[1]-pos2[1])/(pos1[0]-pos2[0])


g = 9.81
mu = 0.6

def affine(m,p,x):
    return (m*x+p)

def in_circle(center,point, radius):
    if euclidean_dist(center,point)<radius**2:
        return True
    else:
        return False
    
def is_between(m,top_p,bot_p,point):
    x = point[0]
    y = point[1]
    if (affine(m,bot_p,x) < y < affine(m, top_p, x)):
        return True
    else:
        return False
    
def angle(point1,point2):
    angle = math.atan(slope(point1,point2))
    if point2[0]>point1[0]:
        return angle
    else:
        return (pi-angle)




class Ball2:
    def __init__(self, pos, colour):
        
        #position is already scaled
        self.pos = pos
        self.colour = colour
        self.r = 2.8
        self.res = info["resolution"]
        
        
    def pixels(self):
        # assign the pixels to the ball , radius = 5.7
        r = round(self.res*self.r)
        x = self.pos[0]
        y = self.pos[1]
        pixels = [[x+r,y], [x-r, y], [x, y+r], [x, y-r], [x+round(r/math.sqrt(2)), y + round(r/math.sqrt(2))], [x-round(r/math.sqrt(2)), y + round(r/math.sqrt(2))],[x+round(r/math.sqrt(2)), y - round(r/math.sqrt(2))],[x - round(r/math.sqrt(2)), y - round(r/math.sqrt(2))]]
        return pixels
        
    def hole_dist(self, hole_pos):
        return math.sqrt(euclidean_dist(self.pos,hole_pos))
    
    def cue_dist(self, cue_ball):
        return math.sqrt(euclidean_dist(self.pos, cue_ball.pos))
    
    def get_angle(self, cue_ball, hole_pos):
        m1 = slope(self.pos, cue_ball.pos)
        m2 = slope(hole_pos, self.pos)
        return math.atan(abs((m2-m1)/(1+m1*m2)))
    
    def initial_force(self, cue_ball, hole_pos):
        angle = self.get_angle(cue_ball, hole_pos)
        initial_force = 2*mu*g*(self.cue_dist(cue_ball)+(2*mu*g*self.hole_dist(hole_pos))/(math.sin(angle/2))**2)
        return initial_force
    
    def actual_target(self,cue,hole_pos):
        r = self.r*self.res
        angle2 = angle(self.pos,hole_pos)
        target = [self.pos[0]-2*r*math.cos(angle2), self.pos[1]-2*r*math.sin(angle2)] # does this work with signs?
        target = [round(target[0]), round(target[1])]
        
        return target
    
    def is_possible(self, cue_ball, hole_pos, balls, hole_bounds): #balls is a list of all balls except the two concerned
        #find actual equation of shot (with adjustment)
        # 1: find adjusted target
        r = self.r*self.res
        angle2 = angle(self.pos,hole_pos)
        target = self.actual_target(cue_ball,hole_pos)
        
        #find equation
        m = slope(cue_ball.pos, target)
        p = m*target[0] + target[1]
        
        #find boundaries
        angle1 = angle(cue_ball.pos, target)+0.01
        shift1 = abs(2*r/math.sin(angle1))
        top_p = p + m*shift1
        bot_p = p - m*shift1
        
        #iterations for checking
        #check cue to ball
        for ball in balls:
            for point in ball.pixels():
                if is_between(m, top_p, bot_p, point) and is_between(0,max(cue_ball.pos[0],ball.pos[0]), min(cue_ball.pos[0],ball.pos[0]), point):
                    return False
                
        #check ball to hole
        #find new line
        
        M = slope(self.pos, hole_pos)
        P = M*hole_pos[0] + hole_pos[1] 
        
        shift2 = abs(2*r/math.sin(angle2))
        top_P = P + M*shift2
        bot_P = P - M*shift2
        
        #check ball to hole
        for ball in balls:
            for point in ball.pixels():
                if is_between(M, top_P, bot_P, point) and is_between(0,max(ball.pos[0],hole_pos[0]), min(ball.pos[0],hole_pos[0]), point):
                    return False
                
        # narrower boundaries when we check the pocket boundaries
        shift3 = shift2/2
        top_P = P + M*shift3
        bot_P = P - M*shift3
        for hole in hole_bounds:
            for point in hole_bounds[hole]:
                if is_between(M, top_P, bot_P, point) and is_between(0,max(ball.pos[0],hole_pos[0]), min(ball.pos[0],hole_pos[0]), point):
                    return False
            
        return True
    

def h_bounds(info):
    s = info["resolution"]*info["corner_len"]
    r = info["resolution"]*info["side_len"]
    [wdt,lth] = info["dimensions"]
    hole_bounds = {
        "1" : [[s,0],[0,s]],
        "2" : [[1,lth//2-r],[1,lth//2+r]],
        "3" : [[s,lth-1],[1,lth-s]],
        "4" : [[wdt-s,lth-1],[wdt-1,lth-s]],
        "5" : [[lth//2-r,wdt-1],[lth//2+r,wdt-1]],
        "6" : [[wdt-s,1],[wdt-1,s]]
    }
    
    return hole_bounds



#create dic of target points for pocket
def t_points(info):
    s = info["resolution"]*info["corner_len"]
    r = info["resolution"]*info["side_len"]
    res = info["resolution"]
    
    
    
    [wdt,lth] = info["dimensions"]
    target_points = [[0 for i in range(3)] for j in range(6)]
    target_points = {
        "1" : [[s//4,(3*s)//4],[s//2,s//2], [(3*s)//4,s//4]],
        "2" : [[1,lth//2-r//2],[1,lth//2],[1,lth//2+r//2]],
        "3" : [[s//4,lth-(3*s)//4],[s//2,lth-s//2],[(3*s)//4,lth-s//4]],
        "4" : [[wdt-(3*s)//4,lth-s//4],[wdt-s//2,lth-s//2],[wdt-s//4,lth-(3*s)//4]],
        "5" : [[wdt-1,lth//2-r//2],[wdt-1,lth//2],[wdt-1,lth//2+r//2]],
        "6" : [[wdt-(3*s)//4,s//4],[wdt-s//2,s//2], [wdt-s//4,(3*s)//4]],
    }
    
    return target_points


hole_bounds = h_bounds(info)
target_points = t_points(info)



import scipy.integrate as integrate
import scipy.special as special
from scipy.stats import *
from pynverse import inversefunc
from math import *
import numpy as np
import matplotlib.pyplot as plt

def prob(ball_r,balls_dist,angle_1,angle_2,cue_ball_angle,omega):

    min_in_angle = min(angle_1,angle_2)
    max_in_angle = max(angle_1,angle_2)

    def f(t):
        try:
            return asin(balls_dist*sin(t)/(2*ball_r)) - abs(t) + cue_ball_angle 
        except:
            pass
    
    def pdf(x):
        var = 0.1 #* exp(-(balls_dist-ball_r*2))
        return pow(2*pi*var,-0.5) * exp(-((x-omega)**2)/(2*var))
    
    try:
        bound = asin(2*ball_r/balls_dist)
    except:
        return 0

    f_image = [min_in_angle,max_in_angle]
    inv_f = inversefunc(f,domain=[-1*bound,bound], image = f_image)


    return integrate.quad(pdf,inv_f(f_image[0]),inv_f(f_image[1]))[0]


#finds the integration bounds
def target_angle_bounds(ball, hole_nb):
    point1, point2 = hole_bounds[hole_nb][0], hole_bounds[hole_nb][1]
    angle1, angle2 = angle(ball.pos,point1), angle(ball.pos, point2) 
    return (angle1,angle2)
   



#For a given cue and colour, iterate the list

def find_optimal_direct(balls,cue,colours):
    shots = []
    radius = balls[0].r*balls[0].res
    
    for ball in balls:
        if ball.colour in colours:
            for hole in target_points:
                (angle1, angle2) = target_angle_bounds(ball,hole)
                cue_angle = angle(cue.pos,ball.pos)
                
                for hole_point in target_points[hole]:
                    
                    target = ball.actual_target(cue, hole_point)
                    
                    omega = cue_angle - angle(target,hole_point)
                    if ball.hole_dist(hole_point)<=cue.hole_dist(hole_point):
                        
                        proba = prob(radius, ball.cue_dist(cue), angle1, angle2, cue_angle, omega)
                    else:
                        proba = 0
                    
                    shots.append([ball,hole,hole_point,proba])
                 
    shots.sort(key=lambda x : x[3], reverse=True)

    
    for shot in shots:
        ball = shot[0]
        hole_point = shot[2]
        if ball.is_possible(cue,hole_point,balls,hole_bounds):
            if shot[3] > 0:
                optimal_shot = {
                    "cue position" : cue.pos,
                    "ball position" : ball.pos,
                    "collision point" : ball.actual_target(cue, hole_point),
                    "target hole" : hole_point,
                    "probability" : shot[3],
                    "bounce" : False
                } 

                return optimal_shot
            
            
        
    return "no available shot"



def check_bounce_shot(balls,cue,colours):
        
    [wdt,lth] = info["dimensions"]        
    
    def Y(x,a):
        return (a*y+b*x)/(a+x)
    
    def X(y,b):
        return (x*y+a*b)/(b+y)
    
    for ball in balls:
        if ball.colour in colours:
            [x,y] = cue.pos
            [a,b] = ball.pos
            cue1 = Ball2([1,Y(x, a)],"white")
            cue2 = Ball2([1,Y(wdt - x, wdt - a)],"white")
            cue3 = Ball2([X(y,b),1],"white")
            cue4 = Ball2([X(lth-y, lth-b),1],"white")
            
            optimals = [find_optimal_direct(balls,cue1,colours), find_optimal(balls,cue2,colours), find_optimal(balls,cue3,colours), find_optimal(balls,cue4,colours),]
            
            optimals.sort(key = lambda x: x["probability"], reverse=True)
            optimals[0]["bounce"] = True
            return optimals[0]



def find_optimal(balls,cue,colours):
    direct_best = find_optimal_direct(balls,cue,colours)
    if direct_best == "no available shot":
        return check_bounce_shot(balls,cue,colours)
    else:
        return direct_best



#if white ball can be placed
def place_white(balls,colour):
    radius = round(balls[0].r*balls[0].res)
    [wdt,lth] = info["dimensions"]
    optimals = []
    for delta_x in range(radius,wdt-radius,4*radius):
        for delta_y in range(radius,lth//4-radius,4*radius):
            cue = Ball2([delta_x,delta_y],"white")
            optimals.append(find_optimal(balls,cue,colour))
    optimals.sort(key = lambda x: x["probability"], reverse=True)
    return optimals[0]





#===============================================================================
# 
# #For each turn
# colour = input() # can be ["red","yellow"] if table is open
# white_on_table = bool(input())
#===============================================================================
def turn(colours, ball_coords):
    balls = []
    cue = None
    for ball in ball_coords:
        balls.append(Ball2(ball[0],ball[1]))
        print(ball[1])
        if ball[1] == 'white':
            cue = Ball2(ball[0],ball[1])
            print('executed')
    #balls is a list of elements of the class ball
    print(list(ball.pos for ball in balls))
    if cue != None:
        return find_optimal(balls,cue,colours)
    else:
        return place_white(balls,colours)






#-------------------------------------------------------------------------------

class MyApp(App):

    def build(self):
        return MainScreen()


if __name__ == '__main__':
    MyApp().run()