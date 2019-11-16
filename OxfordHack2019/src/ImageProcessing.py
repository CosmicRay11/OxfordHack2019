'''
Created on 16 Nov 2019

@author: George
'''

import PIL
import cv2 as cv
import numpy as np

class ImageProcessor(object):


    def __init__(self, url):
        
        self.initial = cv.imread(url)
        
        filteredIm = self.filter_green(self.initial)
        
        filteredIm =  self.resize(filteredIm, 0.2)
        cv.imshow("filtered_green", filteredIm)
        
        
        
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        
    
    def filter_green(self, im):
        (b,g,r) = cv.split(im)
        
        m = np.maximum(np.maximum(b,g), r)
        
        b[True] = 0
        g[g<m] = 0
        r[True] = 0
        
        newIm = cv.merge([b,g,r])
        
        return newIm

    def resize(self, im, factor):
        return cv.resize(im, (0,0), fx=factor, fy=factor)
    
    

if __name__ == "__main__":
    defUrl = "C:\\Users\\George\\Documents\\University\\Hack prep\\2019-11\\IMG_20191112_190928.jpg"
    i = ImageProcessor(defUrl)
    
    
        
        