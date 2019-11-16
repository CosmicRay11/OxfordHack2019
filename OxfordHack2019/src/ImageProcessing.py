'''
Created on 16 Nov 2019

@author: George
'''

import random
import PIL
import cv2 as cv
import numpy as np
from cv2 import line

class ImageProcessor(object):


    def __init__(self, url):
        
        self.baseUrl = "C:\\Users\\George\\Pictures\\Hack_tests"
        
        self.initial = cv.imread(url)
        
    def extract_circles(self, im, tableWidth):
        
        #expected = 0.05 * tableWidth
        expected = 75
        
        circleImage = im.copy()
        
        im = self.filter_for_balls(im)
        #self.show_image("filtered", im)
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        #im = cv.Canny(im, 100, 200)
        
        #im = cv.bitwise_not(im)
        #kernel = np.ones((10,10),np.float32)/4
        #im = cv.filter2D(im,-1,kernel)
        #self.show_image("blur", blurred)
        self.show_image("cannied", im)
        circles = cv.HoughCircles(im,cv.HOUGH_GRADIENT,5,100,
                            param1=60,param2=30,
                            minRadius=int(expected*0.2),maxRadius=int(expected*1.5))
        try:
            circles = np.uint16(np.around(circles))
            print(circles)
            for i in circles[0,:]:
                cv.circle(circleImage,(i[0],i[1]),i[2],(0,0,0),2)
                cv.circle(circleImage,(i[0],i[1]),2,(0,0,0),3)
        except:
            pass
        self.show_image("circle image", circleImage)
    
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
                    print(theta)
                    hLines.append([(m, c)])
                else:
                    print(theta, 'v')
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
    
        self.show_image("line image" + str(random.randint(0,1000)), imCopy)

    def display_xy_lines(self, im, lineList):
        imCopy = im.copy()
        for line in lineList:
            for (m,c) in line:
                x1 = -10000
                y1 = int(m*x1 + c)
                x2 = 10000
                y2 = int(m*x2 + c)
            
            cv.line(imCopy,(x1,y1),(x2,y2),(0,0,255),5)
    
        self.show_image("line image" + str(random.randint(0,1000)), imCopy)
    

    def resize(self, im, factor):
        return cv.resize(im, (0,0), fx=factor, fy=factor)
    
    

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
        i.extract_circles(cutBoard, 2500)
        
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        