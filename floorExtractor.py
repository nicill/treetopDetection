import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy import stats
import sliding_window as sw

#given a window, compute its average
def averageWindow(window): return np.average(window)

def whitePixPerc(img):
    return 100*(np.sum(img == 255)/(img.shape[0]*img.shape[1]))

# Function to erase regions smaller than a given number of pixels or not fat enough
def eraseSmallRegions(im,numPixTh,oneTreePixels=10000):

    retVal=im.copy()

    mask = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY)[1]

    #compute connected components
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

    print("computed connected components, found "+str(numLabels)+" image shape "+str(labelImage.shape))

    #traverse all labels but ignore label 0 as it contains the background
    for l in range(1,numLabels):
        #count pixels with this label
        currentCount=np.count_nonzero(labelImage==l)


        if currentCount<numPixTh:retVal[labelImage==l]=0
        else:

            #print("For label "+str(l)+" there are "+str(currentCount)+" pixels")
            #print("with stats "+str(stats[l]))
            left=stats[l][0]
            top=stats[l][1]
            fatSize=min(stats[l][2],stats[l][3])

            # retrieve the number of non-black pixels in the square
            subImage=labelImage[top:top+fatSize,left:left+fatSize]
            labelPixelsFatRegion=np.count_nonzero(subImage==l)

            #print("found pixels in fat region: "+str(labelPixelsFatRegion))

            #if the region has too few pixels, erase it
            if labelPixelsFatRegion<oneTreePixels:retVal[labelImage==l]=0


    return retVal

def pureGradient(img):
    sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    sobelImg=sobelx+sobely

    cutoff=3
    sobelImg[sobelImg>cutoff]=255
    sobelImg[sobelImg<=cutoff]=0

    return sobelImg

def gradientClosed(dem,closeSize,closeIt):
    img = dem
    #blur
    gray=cv2.GaussianBlur(dem,(5,5),0)

    sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    sobelImg=sobelx+sobely

    cutoff=3
    sobelImg[sobelImg>cutoff]=255
    sobelImg[sobelImg<=cutoff]=0

    #erode and close
    erosionKernel=np.ones((5,5),np.uint8)
    erosion=cv2.erode(sobelImg,erosionKernel,iterations=1)

    closingKernel=np.ones((closeSize,closeSize),np.uint8)
    closing=cv2.morphologyEx(erosion,cv2.MORPH_CLOSE,closingKernel,iterations=closeIt)

    return closing

def joinBlackRegions(listOfMasks):
    if len(listOfMasks)<1: raise Exception("floorExtractor: joinBlackRegions, empty list")
    retVal=listOfMasks[0].copy()
    for newMask in listOfMasks[1:]:retVal[newMask==0]=0
    return retVal

def thresholdLowPixels(dem,th):
    #print("thresholding with "+str(th))
    retVal=dem.copy()
    retVal[dem<th]=0
    retVal[dem>=th]=255
    return retVal

def filterOutShadows(im,tops,th=100):

    # image is read in grayscale
    #print("thresholding shadows "+str(th))
    im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # blur
    gray = cv2.GaussianBlur(im,(5,5),5)

    # Threshold
    thresholded=gray.copy()
    thresholded[gray<=th]=0
    thresholded[gray>th]=255

    #erode
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(255-thresholded,kernel,iterations = 3)
    closeSize=10
    closeIt=3
    kernelClosing = np.ones((closeSize,closeSize),np.uint8)
    close= cv2.morphologyEx(erosion,cv2.MORPH_CLOSE,kernelClosing,iterations = closeIt)

    retVal=255-close

    # Debugging purposes
    #retVal[retVal==255]=200
    #retVal[tops==0]=255

    return retVal

def filterOutColor(im,th):
    # image is read in color
    im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # blur
    gray = cv2.GaussianBlur(im,(5,5),5)

    # Threshold
    thresholded=gray.copy()
    thresholded[gray<=th]=255
    thresholded[gray>th]=0

    #erode
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(255-thresholded,kernel,iterations = 3)
    closeSize=10
    closeIt=7
    kernelClosing = np.ones((closeSize,closeSize),np.uint8)
    close= cv2.morphologyEx(erosion,cv2.MORPH_CLOSE,kernelClosing,iterations = closeIt)

    retVal=255-close

    # Debugging purposes
    #retVal[retVal==255]=200
    #retVal[tops==0]=255

    return retVal


def extendFloor(dem,floorImage,th,wSize):
    # First, slide a window
    for (x, y, window) in sw.sliding_window(dem, stepSize=wSize, windowSize=(wSize, wSize)):
        #print(window.shape)
        # Take also binary floor image window
        floorWindow=floorImage[y:y + wSize, x:x + wSize]
        #compute average height of floor points
        # mask out non floor points
        floorDEM=window.copy()
        floorDEM[floorWindow==255]=0

        # IGNORE ZEROS!
        try:
            avFloorHeight=stats.tmean(floorDEM,(1,255))
        except:
            avFloorHeight=0
        #otherAv=np.average(window)
        #print(str(avFloorHeight)+" "+str(otherAv))
        #Now, make black all point within the threshold of the average floor height
        #print("in this window erasing from "+str(avFloorHeight)+" to "+str(avFloorHeight+th))
        floorWindow[window<(avFloorHeight+th)]=0
    return floorImage

#divide a DEM image into different density parts according to the differences in gradients.
def quantizeImage(dem, tops, floorTh,topTh,closeSize,closeIt):

    image=dem

    #First Crude threshold
    heightLabelImage=image.copy()
    heightLabelImage[image>topTh]=2
    heightLabelImage[image<topTh]=1
    heightLabelImage[image<floorTh]=0

    # For visualization
    #image[image<floorTh]=0
    #image[image>topTh]=200
    #image[tops==0]=255
    #cv2.imwrite("crudeFloor.jpg",image)

    gray = cv2.GaussianBlur(image,(5,5),0)
    laplace=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    laplace2=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    gradientIm=laplace + laplace2
    cutOff=50
    gradientIm[gradientIm<cutOff]=0
    gradientIm[gradientIm>cutOff]=255
    #gradientIm[tops==0]=255
    cv2.imwrite("gradientThresholded.jpg",gradientIm)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gradientIm,kernel,iterations = 1)
    #erosion[tops==0]=255
    cv2.imwrite("gradientThresholdedEroded.jpg",erosion)

    kernelClosing = np.ones((closeSize,closeSize),np.uint8)

    close= cv2.morphologyEx(erosion,cv2.MORPH_CLOSE,kernelClosing,iterations = closeIt)
    gradientLabelImage=close.copy()
    gradientLabelImage[close==255]=2
    gradientLabelImage[close<255]=0
    #0 height is zero!
    gradientLabelImage[heightLabelImage==0]=0

    labelImage=heightLabelImage+gradientLabelImage
    #labelImage=gradientLabelImage

    representLabels=labelImage*61
    #representLabels[tops==255]=255
    cv2.imwrite("viewLabels.jpg",representLabels)

    return labelImage


def main(argv):
    # Receive DEM image in argv[2] and top image in argv[3],
    # then the size of the window in argv[4], and the crude floor threshols in argv[5]
    image=cv2.imread(argv[1],cv2.IMREAD_COLOR)
    dem=cv2.imread(argv[2],cv2.IMREAD_GRAYSCALE)
    tops=cv2.imread(argv[3],cv2.IMREAD_GRAYSCALE)
    mode=int(argv[4])

    if mode==1: # quantize image
        floorTh=int(argv[5])
        topTh=int(argv[6])
        closeSize=int(argv[7])
        closeIt=int(argv[8])
        labelIm=quantizeImage(image,tops,dem,tops,floorTh,topTh,closeSize,closeIt)
        cv2.imwrite(arv[9],labelIm)
    elif mode==2:
        thresholdFloor=int(argv[5])
        thresholdColor=int(argv[6])
        thresholdShadows=int(argv[7])
        extensionThreshold=int(argv[8])
        windowSize=int(argv[9])
        newDEMFile=argv[10]

        #first, threshold low pixels
        lowP=thresholdLowPixels(dem,thresholdFloor)
        lowP2=lowP.copy()
        lowP2[tops==0]=100
        cv2.imwrite("lowPixels.jpg",lowP2)

        #compute shadows
        shadowsBinaryImage=filterOutShadows(image,tops,thresholdShadows)
        shadowsBinaryImage2=shadowsBinaryImage.copy()
        shadowsBinaryImage2[tops==0]=100
        cv2.imwrite("shadows.jpg",shadowsBinaryImage2)

        # Threshold by color
        colorFiltered=filterOutColor(image,thresholdColor)
        colorFiltered2=colorFiltered.copy()
        colorFiltered2[tops==0]=100
        cv2.imwrite("colorFloor.jpg",colorFiltered2)

        allFloor=joinBlackRegions([lowP,shadowsBinaryImage,colorFiltered])

        extendedFloor=allFloor.copy()
        numExtIterations=5
        listIterations=[windowSize,int(windowSize*1.5),int(windowSize*2)]
        for i in range(numExtIterations):
            extendFloor(dem,extendedFloor,extensionThreshold,listIterations[i%len(listIterations)])

        allFloor[tops==0]=100
        cv2.imwrite("allFloor.jpg",allFloor)

        #extendedFloor[tops==0]=100
        cv2.imwrite("ExtendedFloor.jpg",extendedFloor)

        #now filteroutFloor from the DEM
        newDem=dem.copy()
        newDem[extendedFloor==0]=0
        newDem[tops==0]=255
        cv2.imwrite(newDEMFile,newDem)
    elif mode==3:
        cv2.imwrite("removedSmall.jpg",eraseSmallRegions(dem,int(argv[5]),int(argv[6])))





if __name__ == '__main__':
    main(sys.argv)
