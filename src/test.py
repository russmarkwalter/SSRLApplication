import skimage as ski
import matplotlib.pyplot as plot
import numpy as np
import os

IMG_NAME = "chess1"

#import the image
def loadImage(name, num):
    path = "images\\data\\"+name+"\\im"+str(num)+".png"
    return ski.io.imread(path)

#display the image
def showImage(img):
    plot.imshow(img)
    plot.axis("off")
    plot.show()
    
def ssd(arr1, arr2):
    return np.sum((arr1-arr2)**2)



#ASSUMES: epipolar lines are horizontal and aligned
#Currently assumes but doesn't need to assume that resolutions are equal (if one res is lower ig just pretend the pixels are bigger on the columns?)
#quadruple nested four loop; father forgive me. This is horribly inefficient and my second draft will hopefully do better. 
# I can't even run this on a normal sized image. By my estimate, the runtime for a 1920x1080 image is a litte over a week :(. I will clean this up and make another version that I can actually run.
# I am happy that this is very readable though, in my opinion. I would say this first attempt was successful in teaching me some concepts. And the principles should work if I can clean up the for loops.
def correspondanceSearch(im0, im1):
    rows = im0.shape[0]
    cols = im0.shape[1]
    dispMap = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            leftImageSlide = im0[:, j]
            min = 99999999999999999
            #disparity
            disp = 0
            for k in range(cols):
                rightImageSlide=im1[:, k]
                sum=0
                for l in range(rows):
                    #compute SSD
                    sum = ssd(rightImageSlide[l],leftImageSlide[l])
                if(sum<min):
                    min = sum
                    #update disparity if new best column is found
                    disp = abs(j-k)
            dispMap[i, j]=disp
            print("Currently At: " + str((i*rows+j))+"/"+str(rows*cols))
    return dispMap





#Okay, i just retraced my code for the first one and realized some flaws. For example, if we assume what we assumed, each pixel in the column has the same disparity, which is definitely wrong.
#for each pixel, construct a reference box. Then, for the right image, construct a larger box around the coordinates of the pixel ont he left image
#then search that larger box for a match to the reference box. Bababooey.
#This might delve back into for loop hell though, so I will have to think of some clever ways to reuse calculations.
#After testing, runtime is a couple minutes which is much better. I am happy with this being my final version of a disparity map for the "by hand" portion, and I will now look to develop the 3d point map
#Before doing that, however, I will consult the genAIs to see if they can clean this up and get the runtime lower. Just for testing purposes.
def fastCorrespondenceSearch(im0, im1, referenceBoxRadius=3, searchBoxRadius=15):
    rows = im0.shape[0]
    cols = im1.shape[1]

    dispMap = np.zeros((rows, cols))
    #don't loop around the edges to avoid out of bounds
    for i in range(referenceBoxRadius, rows-referenceBoxRadius):
        for j in range(referenceBoxRadius, cols-referenceBoxRadius):
            print("Currently at: ("+str(i)+", "+str(j)+")")
            #slice left reference box out of image
            leftReferenceBox = im0[i-referenceBoxRadius:i+referenceBoxRadius+1, j-referenceBoxRadius: j+referenceBoxRadius+1]
            
            minSSD=99999999
            disp=0
            #need to prevent out of bounds with max/min
            beginSearchWindow = max(j-searchBoxRadius, referenceBoxRadius)
            endSearchWindow = min(j+searchBoxRadius+1, cols - referenceBoxRadius)
            #loop through search zone (1d right now, searches a box along i)
            for k in range(beginSearchWindow, endSearchWindow):
                rightReferenceBox = im1[i-referenceBoxRadius:i+referenceBoxRadius+1, k-referenceBoxRadius:k+referenceBoxRadius+1]
                sum = ssd(leftReferenceBox, rightReferenceBox)
                if(sum<minSSD):
                    minSSD = sum
                    disp = abs(j-k)
            dispMap[i, j]=disp
    return dispMap




            


def main():
    im0 = loadImage(IMG_NAME, 0)
    im1 = loadImage(IMG_NAME, 1)
    im0 = ski.color.rgb2gray(im0)
    im1 = ski.color.rgb2gray(im1)

    dispMap = fastCorrespondenceSearch(im0, im1)

    showImage(dispMap)
    
main()
