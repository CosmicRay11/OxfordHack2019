
@package ImageProcessing
'''
Created on 16 Nov 2019

@author: George
'''

import numpy as np
import ImageProcessor



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
        
        cv.waitKey(0)
        cv.destroyAllWindows()
            