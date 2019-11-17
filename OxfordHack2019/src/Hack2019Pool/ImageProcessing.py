'''
Created on 16 Nov 2019
@author: George
'''

import random
import PIL
import cv2 as cv
import numpy as np
from cv2 import line

import matplotlib
import matplotlib.pyplot as plt

class ImageProcessor(object):


    def __init__(self, url):
        
        self.baseUrl = "C:\\Users\\George\\Pictures\\Hack_tests"
        
        self.initial = cv.imread(url)
    
    def label_balls(self, im, tableWidth):
        imcopy = im.copy()
        
        circles, whiteBall, blackBall = self.extract_circles(im, tableWidth)
        valids = []
        
        #self.show_image("label", im)
        cv.imwrite("C:\\Users\\George\\Pictures\\Hack_tests\circle_stuff.jpg", im)
        
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
        return allBalls
        
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
        cv.imwrite(self.baseUrl + "\refiltered.png", refilteredIm)
        
        edgesIm = self.do_edge_detection(refilteredIm)
        #self.show_image('edges2', edgesIm)
        
        lineList = self.get_lines(edgesIm)
        #self.display_lines(self.initial, lineList)
        self.display_xy_lines(self.initial, lineList)
        
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
    
        xList = []
        yList = []
        for ball in balls:
            x,y,rad,type = ball
            x1 = (y-lines[0][0][1])/lines[0][0][0]
            x2 = (y-lines[1][0][1])/lines[1][0][0]
            dist = abs (x1-x2)
            xratio = (x-x1) / dist
            
            yratio = 1 - (y / image.shape[1])**0.4
            
            #self.project(image, x,y, lines)
            
            print('x',xratio)
            print('y',yratio)
        
            xList.append(yratio*2)
            yList.append(xratio)
             
        xList = np.array(xList)
        yList = np.array(yList)
         
        fig, ax = plt.subplots(figsize=(12,6))
         
        plt.plot(yList, xList, marker='x', color='black', linestyle='None', markersize = 5.0)
         
        axes = plt.gca()
        axes.set_xlim([0,1])
        axes.set_ylim([0,2])
         
         
         
        plt.show()
    
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
        cv.imshow("im", cv.resize(image.copy(), (0,0), fx=0.2, fy=0.2))
        
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
        cv.imshow("im2", cv.resize(image.copy(), (0,0), fx=0.2, fy=0.2))
        return halfWayMark
        
    def rotate_coords(self, coords, origin, radians):
        x, y = coords
        ox, oy = origin
    
        qx = ox + np.cos(radians) * (x - ox) + np.sin(radians) * (y - oy)
        qy = oy + -np.sin(radians) * (x - ox) + np.cos(radians) * (y - oy)
    
        return qx, qy
    

if __name__ == "__main__":
    defUrl = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191115_191414.jpg"
    defUrl2 = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191116_100356.jpg"
    defUrl3 = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191115_191217.jpg"
    defUrl4 = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191115_191345.jpg"
    urls = [defUrl, defUrl2, defUrl3, defUrl4]
    #random.shuffle(urls)
    for url in urls:
        i = ImageProcessor(url)
        lines = i.extract_board()
        cutBoard = i.cut_board(lines)
        i.show_image("cut board", cutBoard)
        balls = i.label_balls(cutBoard, 2500)
        
        proj = Projector(balls, cutBoard, url)
        
        cv.waitKey(0)
        cv.destroyAllWindows()