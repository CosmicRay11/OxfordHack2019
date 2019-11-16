'''
Created on 16 Nov 2019

@author: George
'''

import PIL
import cv2 as cv
import numpy as np

class ImageProcessor(object):


    def __init__(self, url):
        
        self.baseUrl = "C:\\Users\\George\\Pictures\\Hack_tests"
        
        self.initial = cv.imread(url)
        
        greenIm = self.filter_green(self.initial)
        
        smallGreenIm = self.resize(greenIm, 0.2)
        #self.show_image("filtered_green", smallGreenIm)
        
        whiteFreeIm = self.remove_white(greenIm)
        #self.show_image('white filtered', whiteFreeIm)
        
        refilteredIm = self.exclude_rb(whiteFreeIm)
        #self.show_image('refiltered', refilteredIm)
        cv.imwrite(self.baseUrl + "\refiltered.png", refilteredIm)
        
        edgesIm = self.do_edge_detection(refilteredIm)
        #self.show_image('edges2', edgesIm)
        
        lineList = self.get_lines(edgesIm)
        self.display_lines(self.initial, lineList)
        
        print(lineList)
        
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    def show_image(self,label, im):
        print(im.shape)
        width, height = im.shape[0], im.shape[1]
        print(width, height)
        if width > 800:
            factor = 800/width
            print(factor)
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
        #                     target,   rho, theta, something else?
        lines = cv.HoughLines(edgeImage,1,np.pi/180,200)
        return lines
        
    
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
            cv.line(imCopy,(x1,y1),(x2,y2),(0,0,255),2)
        
        self.show_image("line image", imCopy)

    def resize(self, im, factor):
        return cv.resize(im, (0,0), fx=factor, fy=factor)
    
    

if __name__ == "__main__":
    defUrl = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191115_191414.jpg"
    defUrl2 = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191116_100356.jpg"
    defUrl3 = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191115_191217.jpg"
    defUrl4 = "C:\\Users\\George\\Pictures\\Hack_tests\\IMG_20191115_191345.jpg"
    urls = [defUrl, defUrl2, defUrl3, defUrl4]
    for url in urls:
        i = ImageProcessor(url)
    

        
        