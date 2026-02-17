import skimage as ski
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plot
import numpy as np
import os
import cv2 # used for denoising
import pyvista as pv
from scipy.spatial import cKDTree


IMG_NAME = "curule1"

#import the image
def loadImage(name, num):
    path = "images\\data\\"+name+"\\im"+str(num)+".png"
    return ski.io.imread(path)

def loadCalibration(image_name):
    file = open("images\\data\\"+image_name+"\\calib.txt")
    baseline = 0
    f0 = 0
    f1 = 0
    cx = 0
    cy = 0
    doffs = 0
    for line in file:
        variable, value = line.split("=")
        variable = variable.strip().lower()
        value = value.strip().lower()
        if variable == 'cam0':
            # expects [fx 0 cx0; 0 fy cy0; 0 0 1], so parse into a single array
            numbers = value.replace('[', '').replace(']', '').replace(';', '').split()
            f0 = float(numbers[0])
            f1 = float(numbers[4])   
            cx = float(numbers[2])
            cy = float(numbers[5])
        if(variable=="doffs"):
            doffs=float(value)
        if(variable=="baseline"):
            baseline=float(value)
    file.close()
    return baseline, f0, f1, doffs, cx, cy



#display the image
def showImage(img):
    plot.imshow(img, cmap='gray')
    plot.axis("off")
    plot.show()
    
def ssd(arr1, arr2):
    return np.sum((arr1-arr2)**2)


#FIRST ATTEMPT
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




#SECOND ATTEMPT
#Here is a much better version I cooked up. Runtime is much chiller, as we now only search along the row containing the pixel for a match of the pixel. I can do this because I am assuming parallel and same focus cameras.
#Runtime scales with the search box radius and the square of reference box radius.
def fastCorrespondenceSearch(im0, im1, referenceBoxRadius=3, searchBoxRadius=120):
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
            endSearchWindow = j
            #loop through search zone (1d right now, searches a box along i)
            for k in range(beginSearchWindow, endSearchWindow):
                rightReferenceBox = im1[i-referenceBoxRadius:i+referenceBoxRadius+1, k-referenceBoxRadius:k+referenceBoxRadius+1]
                sum = ssd(leftReferenceBox, rightReferenceBox)
                if(sum<minSSD):
                    minSSD = sum
                    disp = abs(j-k)
            dispMap[i, j]=disp
    return dispMap



#THIRD ATTEMPT
#this was made using a small amount of help from gemini (only the uniform filter line)
#the idea is to calculate all ssds for a given disparity at once, so that they don't need to be repeatedly recalculated. Cool.
def finalCorrespondanceSearch(im0, im1, referenceBoxRadius=3, searchLineLength=120):
    rows, cols = im0.shape
    winSize = 2 * referenceBoxRadius + 1

    dispMap = np.zeros((rows, cols), dtype=np.float32)
    minSSD = np.full((rows, cols), 999999999, dtype=np.float32)

    # We'll only keep results for pixels that are at least 'referenceBoxRadius' 
    # away from the left border, because the left border regions are contaminated 
    # by zero‑padding in im1Shifted. The inner area will be extracted later.
    valid_mask = np.zeros((rows, cols), dtype=bool)
    valid_mask[:, referenceBoxRadius: cols - referenceBoxRadius] = True

    for disparity in range(1, searchLineLength + 1):
        im1Shifted = np.zeros_like(im1)
        # shift right by 'disparity' – leftmost 'disparity' columns become zero
        im1Shifted[:, disparity:] = im1[:, :-disparity]

        squaredDiffs = (im0 - im1Shifted) ** 2
        summedSquaredDiffs = uniform_filter(squaredDiffs, size=winSize) * (winSize ** 2) #GEMINI LINE

        # update only where SSD is lower AND the pixel is in the valid region
        mask = (summedSquaredDiffs < minSSD) & valid_mask
        minSSD[mask] = summedSquaredDiffs[mask]
        dispMap[mask] = disparity
    
    return dispMap



#basic denoising
def processImage(img, kernalSize=3, min_valid_disparity=1):

    processedImage = img.copy().astype(np.float32)
    


    # Median blur to remove salt‑and‑pepper noise
    processedImage = cv2.medianBlur(processedImage, kernalSize)

    # Remove far points: very small disparities → set to 0
    processedImage[processedImage < min_valid_disparity] = 0.0

    return processedImage


#vesitigal
#From Middlebury:
#Z = baseline * f / (d + doffs)
def getDepthMap(dispMap, f, b, doffs):

    depthMap=np.zeros((dispMap.shape[0], dispMap.shape[1]))
    depthMap=depthMap.astype(np.float32)

    for i in range(depthMap.shape[0]):
        for j in range(depthMap.shape[1]):
            depthMap[i, j] = (f*b)/(doffs+dispMap[i, j, 0])
    return depthMap
    
#Uses algorithms from the middlebury website to convert disparities to points in 3d space.
def getPointCloud(dispMap, f, b, doffs, cx, cy, maxDepth=10000):
    rows, cols = dispMap.shape[:2]
    points = np.zeros((rows, cols, 3), dtype=np.float32)

    for v in range(rows):
        for u in range(cols):
            d = dispMap[v, u]
            if d <= 0:                     # skip invalid disparities
                continue

            denom = d + doffs
            if abs(denom) < 1e-6:           # avoid division by zero
                continue

            Z = (f * b) / denom
            if Z > maxDepth:                # discard points beyond max_depth
                continue

            X = (u - cx) * Z / f
            Y = (v - cy) * Z / f
            points[v, u] = [X, Y, Z]

    return points


#this was generated by gemini to display the pointmap. 
def clean_and_display_pointcloud(points, method='statistical', 
                                 nb_neighbors=20, std_ratio=2.0,
                                 voxel_size=None, point_size=2):
    if points.shape[0] == 0:
        print("Point cloud is empty. Nothing to display.")
        return None

    # Optional voxel downsampling (simple grid average)
    if voxel_size is not None and voxel_size > 0:
        # Discretize coordinates into voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(int)
        # Get unique voxel keys
        _, unique_inv = np.unique(voxel_indices, axis=0, return_inverse=True)
        # Average points in each voxel
        points = np.array([points[unique_inv == i].mean(axis=0) 
                           for i in range(unique_inv.max() + 1)])
        print(f"Downsampled to {len(points)} points.")

    # Statistical outlier removal
    if method.lower() == 'statistical' and len(points) > nb_neighbors:
        tree = cKDTree(points)
        # Compute distances to k nearest neighbors (k = nb_neighbors)
        distances, _ = tree.query(points, k=nb_neighbors + 1)  # +1 because point itself is included
        avg_dist = np.mean(distances[:, 1:], axis=1)           # exclude self

        mean_dist = np.mean(avg_dist)
        std_dist = np.std(avg_dist)
        threshold = mean_dist + std_ratio * std_dist

        # Keep points whose average neighbor distance is below threshold
        mask = avg_dist < threshold
        points = points[mask]
        print(f"Statistical outlier removal kept {len(points)} points.")

    elif method.lower() != 'none':
        print(f"Unknown method '{method}'. Displaying raw cloud.")

    # Create PyVista PolyData and plot
    cloud = pv.PolyData(points)
    # Add depth (Z coordinate) as scalar for color gradient
    cloud.point_data["depth"] = points[:, 2]
    # Use reversed colormap so that farther points have different color (gradient reversed)
    cloud.plot(render_points_as_spheres=True, point_size=point_size,
               scalars="depth", cmap="viridis_r", show_scalar_bar=True)

    return cloud


            

def main():
    im0 = loadImage(IMG_NAME, 0)
    im1 = loadImage(IMG_NAME, 1)
    im0 = ski.color.rgb2gray(im0)
    im1 = ski.color.rgb2gray(im1)
    baseline, f0, f1, doffs, cx, cy = loadCalibration(IMG_NAME)

    dispMap = finalCorrespondanceSearch(im0, im1)
    dispMap = processImage(dispMap)
    showImage(dispMap)

    pointMap = getPointCloud(dispMap, f0, baseline, doffs, cx, cy, 30000)

    #This came from Gemini, as I did not know how to effectively visualize this 3d point map. The noise could be cleaned up significantly better, but this is the literal point map produced by the disparity, with minial cleaning.
    #You can recognize most of the main objects in the point map. It struggles with the orange box int he background, but everything else is very clear when viewed in this manner.
    #If You'd like, you can export this pointMap object and play around with it in any software you'd like.
    points_reshaped = pointMap.reshape(-1, 3)
    mask = np.isfinite(points_reshaped).all(axis=1) & (points_reshaped[:, 2] > 0)
    points_clean = points_reshaped[mask]
    points_clean = points_clean[points_clean[:, 2] < 5000]

    if len(points_clean) > 0:
        # Clean and display
        print("Displaying")
        clean_and_display_pointcloud(points_clean, 
                                     method='statistical',
                                     nb_neighbors=20,
                                     std_ratio=2.0,
                                     voxel_size=None,   # or e.g., 0.05 if you want downsampling
                                     point_size=2)
    else:
        print("No valid points to display.")


    

main()