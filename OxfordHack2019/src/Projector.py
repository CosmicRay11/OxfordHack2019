'''
Created on 16 Nov 2019

@author: George
'''

import np

class Projector(object):

    def __init__(self, balls, image, lines):
        
        angle = np.arctan(lines[3][0][0]) * (180/np.pi)
        