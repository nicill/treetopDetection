import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,50))
import cv2

import numpy as np
import sys
from math import sqrt
from scipy import ndimage

def maskOutWrongParts(demName,maskName,outName):
    """
        Receives a dem and a mask showing the parts
        of the dem that contain incorrect information
        Put to zero all the pixels in the DEM that are
        white in the mask
    """
    dem = cv2.imread(demName,cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(maskName,cv2.IMREAD_UNCHANGED)

    # resample if necessary
    if dem.shape != mask.shape:
        mask = cv2.resize(mask, (dem.shape[1],dem.shape[0]), interpolation = cv2.INTER_LINEAR)

    dem[mask>0] = 0
    cv2.imwrite(outName,dem)

def sliding_window(image, stepSize, windowSize, allImage=False):
    if allImage:
        yield(0,0,image[:,:])
    else:
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def resampleMaskAndBoxes(maskFile, boxesFile, shp, outMask, outBoxes):
    """
        Receives a mask, a bounding boxes file
        and a new shape
        Reshapes the image and rewrites the boxes coordinates
        into two ouput files
    """
    # Resize points image
    maskBefore = cv2.imread(maskFile,0)
    shpBefore = maskBefore.shape
    factorx = shpBefore[0]/shp[1]
    factory = shpBefore[1]/shp[0]

    outIm = cv2.resize(maskBefore, shp, interpolation = cv2.INTER_LINEAR)
    # make a black mask, we will get the points from the boxes
    outIm[outIm !=0 ] = 0

    with open(boxesFile,"r") as f:
        with open(outBoxes,"w") as f2:
            for l in f:
                x1,y1,x2,y2 = l.strip().split(" ")
                nx1,ny1,nx2,ny2 = float(x1)/factorx,float(y1)/factory,float(x2)/factorx,float(y2)/factory

                f2.write(str(nx1)+" "+str(ny1)+" "+str(nx2)+" "+str(ny2)+os.linesep)
                cy = int(ny1+(ny2-ny1)/2)
                cx = int((nx1+(nx2-nx1)/2))
                #print(str(cx)+" "+str(cy))
                cv2.circle(outIm, ( cx,cy ), 5, 255, -1)


    cv2.imwrite(outMask,255-outIm)


def fillHolesBinary(im):
    """
    Receive a binary image with rings and
    turn it into a binary annotation image
    """
    # The function expects white information
    # over black background, change
    im = 255 - im
    # This function is for binary values, change 255 for 1
    im[im > 0] = 1
    # call function
    out = ndimage.binary_fill_holes(im).astype(int)
    #change back to 255
    out[out==1] = 255
    # change back to white over black
    return 255 - out

def eraseBorderPixels(dem, borderTh = 5):
    """
    Function that receives a dem and erases the pixels that are closer
    than borderTh to the outside of the mask.
    """
    demMask = dem.copy()
    # create binary mask for the DEM
    #demMask[ dem == 0] = 0
    demMask[ dem < 0] = 0
    demMask[ dem > 0] = 255

    demMask = np.asarray(demMask, dtype='uint8')

    #cv2.imwrite("mask.jpg",demMask)

    # compute the distance transform to the nearest non-mask pixel (in pixels)
    distMask = cv2.distanceTransform(demMask, cv2.DIST_L2, 3)
    #cv2.imwrite("dist.jpg",distMask)
    # now delete the pixels in the border of the Dem
    dem[distMask<=borderTh] = 0
    return dem


def refineTopDict(topsDict,eps):
    """
    Receives a dictionary "topsDict". Every key represents a label that groups tops together tops from the same connected component
    Every value is a list of tuples "topList" with each tuple containing the altitude of a top and its coordinates.
    Receives also an uncertainty value "eps"

    The function goes over the information from tops in each connected component.

    Inside each component, tops closer than 2*epsilon are considered and only the highest is kept
    """

    retDict = {}
    eliminatedSet = set()

    for k,v in topsDict.items():
        # for every list of tops, first sort in descending altitude order
        sortedTops = sorted(v, key = lambda x : x[0],reverse=True)
        #print("**********************************************:sorted tops "+str(sortedTops))

        for i in range(1,len(sortedTops)):
            #look at the tops before me and if any of them is close to me, eliminate me
            for j in range(i):
                if distPoints(sortedTops[i][1],sortedTops[j][1]) < 2*eps:
                    eliminatedSet.add(sortedTops[i][1])
                    #print(str(k)+",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,going to eliminate "+str(sortedTops[i])+" because it was too close to "+str(sortedTops[j]))
                    break

        # now keep only those not eliminated
        retDict[k] = [sortedTops[0]]
        for i in range(1,len(sortedTops)):
            if not sortedTops[i][1] in eliminatedSet:retDict[k].append(sortedTops[i])

    return retDict


def dictToTopsList(topsDict):
    """
    Receives a dictionary "topsDict". Every key represents a label that groups tops together tops from the same connected component
    Every value is a list of tuples "topList" with each tuple containing the altitude of a top and its coordinates.

    The function retrieves all existing tops and returns them in a list.
    """

    topList = []
    for k,v in topsDict.items():
        for x in v:topList.append(x[1])

    return topList


def distPoints(p,q):
    return sqrt( (p[0]-q[0])*(p[0]-q[0])+(p[1]-q[1])*(p[1]-q[1]))


def storePrettyDEM(demOrig,path):
    dem = demOrig.copy()

    #Filter non values and outliers
    dem[dem<0]=0 #eliminate non values
    #dem[dem>50]=0

    # take out the min (sort of)
    demPerc=dem.copy()
    demPerc[demPerc==0]=np.nan
    minDem = np.nanpercentile(demPerc,1)

    dem = dem - minDem
    dem[dem<0] = 0

    maxDem=np.max(dem)

    prettyDem = (dem*(255/maxDem)).astype("uint8")
    cv2.imwrite(path,prettyDem)

    return prettyDem



# Given a window, eliminate possible outliers and get only the top pixels
def binarizeWindowReturnDEM(win, lowerPerc = 20, eroKernS = 2, eroIt = 3):

    stupidCount+=1

    higherPerc = 99

    winRet = win.copy()
    winPerc = win.copy()

    winPerc[winPerc==0]=np.nan
    minWin = np.nanpercentile(winPerc,lowerPerc)
    maxWin = np.nanpercentile(winPerc,higherPerc)
    #print("slidingWindow, binarizeWindow, window height "+str(minWin)+" "+str(minWin+ (maxWin-minWin))+" "+str(maxWin))

    winRet[win>maxWin] = maxWin
    #winRet[win<minWin + (maxWin-minWin)*fromRatio] = 0
    winRet[win<minWin] = 0

    winRet = winRet-minWin
    winRet = winRet*(255/(maxWin-minWin))

    #cv2.imwrite("./shit/"+str(stupidCount)+"noteroded.jpg",winRet.astype("uint8"))

    erosionKernel=np.ones((eroKernS,eroKernS),np.uint8)
    erosion=cv2.erode(winRet,erosionKernel,iterations = eroIt)

    #winRet[winRet>0] = 255

    return erosion

def resampleDemAndMask(fileDem,fileMask, factor):
    """
        Function to resample a tif file and a
        accompaning mask annotation file
    """
    dem = cv2.imread(fileDem,cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(fileMask,0)
    newD = cv2.resize(dem, (int(dem.shape[1]*factor), int(dem.shape[0]*factor)), interpolation = cv2.INTER_LINEAR)
    mask[mask<100]=0
    mask[mask>0]=255
    print(np.sum(mask==0))
    newMask = cv2.resize(mask, (int(mask.shape[1]*factor), int(mask.shape[0]*factor)), interpolation = cv2.INTER_NEAREST)

    newMask[newMask < 100] = 0
    newMask[newMask > 0] = 255
    print(np.sum(newMask==0))

    cv2.imwrite("newDem.tif",newD)
    cv2.imwrite("newMask.png",newMask)

# Given a window, eliminate possible outliers and get only the top pixels
def binarizeWindow(win, stupidCount, lowerPerc = 10, eroKernS = 5, eroIt = 3, debugImages = "NO" ):

    higherPerc = 99

    winRet = win.copy()
    winPerc = win.copy()

    winPerc[winPerc==0]=np.nan
    minWin = np.nanpercentile(winPerc,lowerPerc)
    maxWin = np.nanpercentile(winPerc,higherPerc)
    #print("slidingWindow, binarizeWindow, window height "+str(minWin)+" "+str(maxWin))

    winRet[win>maxWin] = maxWin
    #winRet[win<minWin + (maxWin-minWin)*fromRatio] = 0
    #winRet[win<minWin] = 0

    winRet = winRet-minWin
    winRet[ winRet<0 ] = 0
    winRet = winRet*(255/(maxWin-minWin))

    # EROSION
    erosionKernel=np.ones((eroKernS,eroKernS),np.uint8)
    erosion=cv2.erode(winRet.astype("uint8"),erosionKernel,iterations = eroIt)

    # code to visualize the output of the erosion
    if debugImages != "NO":
        #erosion = winRet
        sambomba1 = cv2.resize(win*(255/maxWin), (win.shape[1]*10, win.shape[0]*10), interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(debugImages+"/"+str(stupidCount)+"noteroded.jpg",sambomba1)
        sambomba2 = cv2.resize(erosion, (erosion.shape[1]*10, erosion.shape[0]*10), interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(debugImages+"/"+str(stupidCount)+"eroded.jpg",sambomba2)

        stupidCount+=1
    #return a binary mask of the part of the image that needs to be checked
    ret = erosion.copy()
    ret[erosion>0]=255
    return erosion,stupidCount
    #return winRet,stupidCount


def main(argv):

    if (int(argv[1]))==0: # read a mask image and a ROI and erase the masked parts from the roi

        roi = cv2.imread(argv[2],0)
        if roi is None:raise Exception("imageUtils no ROI at "+str(argv[2]))

        treeMask = cv2.imread(argv[3],0)
        if treeMask is None:raise Exception("imageUtils no maskTops at "+str(argv[3]))

        roi[treeMask==0]=255

        cv2.imwrite(argv[2],roi)

    else:
        raise Exception("demUtils, wrong code")

if __name__ == '__main__':
    #example of execution for resampling
    #python imageUtils.py ./combination/Dec24/Binary_Mask_epochs_100_IoU_0.1_confid_0.1.png ./combination/Dec24/Cordinates_epochs_100_IoU_0.1_confid_0.1.txt 1152 4384 ./combination/MaskDec24.png ./combination/boxesDec24.txt

    #resampleDemAndMask(sys.argv[1], sys.argv[2],0.25)
    #cv2.imwrite("filledYOLO.png",fillHolesBinary(cv2.imread(sys.argv[1],0)))
    #resampleMaskAndBoxes(sys.argv[1], sys.argv[2], (int(sys.argv[3]),int(sys.argv[4])), sys.argv[5], sys.argv[6])
    maskOutWrongParts(sys.argv[1], sys.argv[2], sys.argv[3])

    #main(sys.argv)
