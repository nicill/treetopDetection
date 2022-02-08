# import the necessary packages
import argparse
import time
import cv2
import os
from matplotlib import pyplot as plt
import sys
import numpy as np
import math

# peak local max imports
from skimage import data, img_as_float
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

# for the function to turn binary files into lists of points
import crownSegmenterEvaluator as CSE
from sklearn.neighbors import KDTree
#import floorExtractor as fe
#import clusteringMethods as cm
import demUtils as ut
from osgeo import gdal

def sliding_window(image, stepSize, windowSize, allImage=False):
    if allImage:
        yield(0,0,image[:,:])
    else:
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def pureGradient(img):

    #cv2.imwrite("aaa.jpg",img)

    #erode
    erosionKernel=np.ones((1,1),np.uint8)
    erosion=cv2.erode(img,erosionKernel,iterations=1)

    #cv2.imwrite("bbb.jpg",erosion)

    sobelx=cv2.Sobel(erosion,cv2.CV_64F,1,0,ksize=3)
    sobely=cv2.Sobel(erosion,cv2.CV_64F,0,1,ksize=3)
    sobelImg=sobelx+sobely

    cutoff=1
    sobelImg[sobelImg>cutoff]=255
    sobelImg[sobelImg<=cutoff]=0

    return sobelImg


def refineSeedsWithMaximums(inpImage,maskImage,refineRadius=40,seeds=None):


    mask = cv2.threshold(255-maskImage, 80, 255, cv2.THRESH_BINARY)[1]
    #compute connected components
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

    #consider only those containing tops
    dictLabels={}
    for thisTop in seeds:
        thisTopLabel=labelImage[thisTop[1],thisTop[0]]
        if not thisTopLabel in dictLabels:dictLabels[thisTopLabel]=True

    #minPercentage=20
    #minArea=int((minPercentage/100)*math.pi*refineRadius*refineRadius)
    #print("refining with maximums, min area: "+str(minArea))

    # For every connected component, extract mask
    #auxMask= np.zeros([inpImage.shape[0],inpImage.shape[1]],dtype=np.uint8)
    coordList=[]
    count=0
    for i in range(1,numLabels): #0 contains background
        if i in dictLabels:
            count+=1
            #instead of doing shit with masks, just look at the proper part of the image
            left=stats[i,cv2.CC_STAT_LEFT]
            top=stats[i,cv2.CC_STAT_TOP]
            width=stats[i,cv2.CC_STAT_WIDTH]
            height=stats[i,cv2.CC_STAT_HEIGHT]
            area=stats[i,cv2.CC_STAT_AREA]
            max=-1
            maxPos=(0,0)

            for x in range(left,left+width):
                for y in range(top,top+height):
                    #print("you")
                    if labelImage[y][x] == i:
                        if inpImage[y][x]>max:
                            #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ "+str(x)+" "+str(y))
                            max=inpImage[y][x]
                            maxPos=(x,y)
                    #else: print("NO RIGHT LABEL ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; "+str(x)+" "+str(y))

            coordList.append((maxPos[0],maxPos[1]))
    #print("entered the loop "+str(count)+" length "+str(len(coordList)))

    return coordList
#lower=-1 means not lower, otherwise contains the value where to start sampling
def findTops(comp,args,minPix,maxPix,lower): #receive a grayscale image of a connected component, threshold it repeatedly until you find all tree tops
    #pixMinConComp=int(args["minPixTop"])
    pixMinConComp=minPix

    nonBlack=np.sum(comp!=0)
    minTrees=nonBlack/maxPix
    maxTrees=nonBlack/minPix

    #print("MAX "+str(maxTrees))
    #print("MIN "+str(minTrees))

    #check that there are enogh gradients to have some trees
    gradientIm=pureGradient(comp)
    gradientPixelPerc=np.sum(gradientIm!=0)/minPix

    #print("gradpixperc "+str(gradientPixelPerc))
    #print("This image should contains between "+str(minTrees)+" and "+str(maxTrees)+" Trees ")
    if nonBlack<minPix:return []

    #loop over different bands of the DEM, from top to bottom
    #if lower!=-1:
    demIstep=float(args["topStep"])
    demIstart=np.max(comp)-demIstep
    demIend=np.min(comp[np.nonzero(comp)])
    demLength=demIstart-demIend
    demNumSteps=demLength/demIstep

    maxIt=int(1*demNumSteps)

    if lower==-1:
        if gradientPixelPerc<0.025:return []
        elif gradientPixelPerc<1:
            demIstep=demIstep*2
    else:#lower!!!!
        if gradientPixelPerc<0.15:
            return []
        elif gradientPixelPerc<1:
            demIstep=demIstep*5
        elif gradientPixelPerc<1.2:
            demIstep=demIstep*2
        else:
            demIstep=demIstep*1.2

    numIterations=1
    finished=False
    numberTopsFoundList=[]
    tops=[]
    aupo=0
    while demIstart+demIstep>demIend:
        # First get the upper band of the DEM
        #cv2.imwrite(str(aupo)+"comp.png",comp)
        #aupo+=1
        thisBand=ut.thresholdDEMBinarised(comp,demIstart,demIstart+demIstep*numIterations)

        erosionKernel=np.ones((1,1),np.uint8)
        thisBand=cv2.erode(thisBand,erosionKernel,iterations=2)

        # compute connected components here
        numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(thisBand)
        #print("this band has this many connected components "+str(numLabels))

        # for every existing top, make a note of its label in this class in a dictionary
        labelDict={}
        for top in tops:
            thisTopLabel=labelImage[top[0],top[1]]
            if thisTopLabel not in labelDict: labelDict[thisTopLabel]=[comp[top[0],top[1]]]
            else: labelDict[thisTopLabel].append(comp[top[0],top[1]])
        #print("Label dictionary "+str(labelDict))

        for label,listOfHeights in labelDict.items():
            # also update the maximum if necessary, and if this happens this is bad
            auxWin=comp.copy()
            auxWin[labelImage!=label]=0

            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(auxWin)

            if maxVal>max(listOfHeights):
                #print("found valley! "+str(maxVal)+" "+str(max(listOfHeights)))
                tops.append((int(maxLoc[1]),int(maxLoc[0])))


        # traverse all current labels (but the background), the centroids of those not in the dictionary get added as tree tops
        candidateTops=[]
        for l in range(1,numLabels):
            currentCount=np.count_nonzero(labelImage==l)
            if currentCount>pixMinConComp and not l in labelDict:
            #if currentCount>pixMinConComp:
                # find the highest point in this connected component
                auxWin=comp.copy()
                auxWin[labelImage!=l]=0

                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(auxWin)
                candidateTops.append((int(maxLoc[1]),int(maxLoc[0])))

        tops.extend(candidateTops)

        #print("now have "+str(len(tops))+" treeTops ")
        demIstart-=demIstep
        numIterations+=1
        numberTopsFoundList.append(len(tops))

        if numIterations>=maxIt :
            #print("Ofinished tops in "+str(numIterations)+", finding "+str(len(tops))+" "+str(numberTopsFoundList)+"\n\n\n")
            return tops


    #if  len(tops)>maxTrees or len(tops)<minTrees:
    #    print("WRONG NUMBER!!!!!!!!!!!!!! was:"+str(len(tops))+" SHOULD HAVE BEEN between "+str(minTrees)+" and "+str(maxTrees)+" Trees ")

    return tops

# isLower = -1 means not lower window, otherwise it contains the value from wich to start
# if isLower != -1 then listOfTops will contain existing tops
def processWindow(win,args,minNumPointsTree,maxNumPointsTree,isLower=-1):
    # binarize image, erode it a little, compute connected connectedComponents
#    nonBlack=np.sum(win>0)
#    if nonBlack>0:
#        print("NON BLACK!!! "+str(nonBlack))
#        cv2.imwrite("binarised.jpg",win)
#        cv2.imwrite("binarised.png",win)

    binarized=win.copy()
    binarized[win<0.001]=0
    binarized[win>=0.001]=255

    #erode
    # TODO the kernel may now be too big
    #erosionKernel=np.ones((5,5),np.uint8)
    #erosion=cv2.erode(binarized.astype("uint8"),erosionKernel,iterations=1)
    #cv2.imwrite("eroded.jpg",erosion)

    # now, for every connected component, find tops
    #numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(erosion)
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(binarized.astype("uint8"))

    seeds=[]

    for l in range(1,numLabels):
        #for each label, count tops
        thisComponent=win.copy()
        thisComponent[labelImage!=l]=0
        # find tree tops only in connected components that may contain a tree
        if np.sum(thisComponent!=0)>minNumPointsTree:
            thisCompTops=findTops(thisComponent,args,minNumPointsTree,maxNumPointsTree,isLower)
            seeds.extend(thisCompTops)

    # maybe refine seeds in the window, if two seed are too close, eliminate one of them (or something like that)
    return seeds

def paintTopsTrimNonCanopy(dem,seeds,circleSize,cutoff):
    numBands=10
    margin=1.5

    erosionKernel=np.ones((5,5),np.uint8)
    maskImage=255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)

    #first, find the lower and upper heights of the tops
    firstSeed=seeds[0]
    lowerTop=dem[firstSeed[1],firstSeed[0]]
    higherTop=dem[firstSeed[1],firstSeed[0]]

    # find higher and lower tops
    for seed in seeds:
        thisTopHeight=dem[seed[1],seed[0]]
        if lowerTop>thisTopHeight:lowerTop=thisTopHeight
        if higherTop<thisTopHeight:higherTop=thisTopHeight

    #now build a list with the tops sorted in bands
    step=(higherTop-lowerTop)/numBands
    bands=[None]*numBands
    for i in range(len(bands)):bands[i]=[]
    #print("built bands"+str(bands))
    #put each top in the band where it goes
    for seed in seeds:
        #print("checking top "+str(seed)+" in "+str(dem.shape))
        thisTopHeight=dem[seed[1],seed[0]]
        position=min(int((higherTop-thisTopHeight)/step),numBands-1)
        #print("appending at "+str(position)+str(bands))
        bands[position].append((seed,thisTopHeight))

    #print("Finished Banding "+str(higherTop)+" "+str(lowerTop) )
    #for i in range(len(bands)):
    #    print("Band "+str(i)+" : "+str(bands[i]))

    bandCount=0
    for x in bands:
        if len(x)>0:
            partialMask=255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)

            # each position of each band contains a tuple (top,height) with top being a tuple itself
            aTop=x[0][0]
            #print("FIRST top in the band check "+str(aTop))
            thisBandLowerTop=x[0][1]
            thisBandHigherTop=x[0][1]
            lowerBand=(thisBandHigherTop<cutoff)
            for thisTop,thisTopHeight in x:
                #print("checking "+str(thisTop)+" "+str(thisTopHeight))
                if thisBandLowerTop>thisTopHeight:thisBandLowerTop=thisTopHeight
                if thisBandHigherTop<thisTopHeight:thisBandHigherTop=thisTopHeight
                #paint lower tops with bigger radius
                #if thisTopHeight<cutoff:lowerBand=True
                cv2.circle(partialMask, thisTop, circleSize, 0, -1)

            #print("This band lower and higher "+str(thisBandLowerTop)+" "+str(thisBandHigherTop))

            #once finished, paint out anything not in the band
            partialMask[dem>thisBandHigherTop]=255
            if lowerBand:  partialMask[dem<(thisBandLowerTop-margin*1.5)]=255
            else:  partialMask[dem<(thisBandLowerTop-margin)]=255

            #now, only keep regions that contain a top!!!!!!
            #compute connected components

            # now, accumulate to the final mask, depending on the band number
            maskImage[partialMask==0]=0

            #cv2.imwrite("partial"+str(bandCount)+".jpg",maskImage)
            bandCount+=1
            #print("margin now "+str(margin))

    #erosion=cv2.erode(255-maskImage,erosionKernel,iterations=3)
    #cv2.imwrite("PartialFinal.jpg",255-erosion)
    return maskImage

def outputImages(dem,highSeeds,seeds,args,cutoff,index=None):

    #If refine, paint the high seeds first and carefully paint the others after
    if (args["refine"] is not None) and (args["refine"]=="yes") and (args["refineRadius"] is not None):

        maskImage=255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)
        #First, paint the high seeds
        #circleSize=40
        #for seed in highSeeds:
            # create binary result image
        #    cv2.circle(maskImage, seed, circleSize, 0, -1)

        #cv2.imwrite("highSeeds.jpg", maskImage)

        #then, refine the lower seeds
        circleSize=int(args["refineRadius"])
        print("CIRCLESIZE "+str(circleSize))

        seeds.extend(highSeeds)

        if(len(seeds)>0):
            #prepare for refinement, disconnect floor and lower regions
            maskImageLow=paintTopsTrimNonCanopy(dem,seeds,circleSize,cutoff)

            #join the two masks
            maskImage[maskImageLow==0]=0

        #cv2.imwrite("REfinal.jpg",maskImage)

        #refine by taking only the highest point in every resulting region and eliminating small regions
        outputSeeds=refineSeedsWithMaximums(dem,maskImage,int(args["refineRadius"]),seeds)
    else:
        print("Skipping seed refinement!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        seeds.extend(highSeeds)
        outputSeeds=seeds
    print("number of refined seeds! "+str(len(outputSeeds)))

    # Once the method is finished, paint the detected points
    # paint them as circles over the DEM and the image
    #print(" Total seeds, found "+str(len(seeds)))

    circleSize=2
    maskImage=255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)
    for seed in outputSeeds:
        cv2.circle(dem, seed, circleSize, (255, 255, 255), -1)
        # also create binary result image
        cv2.circle(maskImage, seed, circleSize, 0, -1)

    # Finally, store all annotated images
    #filename, file_extension = os.path.splitext(args["dem"])
    #out_dem = filename + "_marked" + file_extension
    filename, file_extension = os.path.splitext(args["dem"])
    if args["binOut"] is not None: out_binary = args["binOut"]
    else: out_binary = filename + "_binaryMethodCCHEIGHT"+ file_extension

    if index is not None: out_binary=out_binary[:-4] +str(index)+ out_binary[-4:]
    print("out binary will be "+out_binary)

    #cv2.imwrite(out_dem, dem)
    cv2.imwrite(out_binary, maskImage)


#simple function to just eliminate lower seeds already found in the higher band
def refineLower(dem,seeds,lowerSeeds,args):

    #eliminate those that were already in the initial list of seeds
    maskImage=255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)

    circleSize=3

    # create mask image
    for seed in seeds:
        cv2.circle(maskImage, seed, circleSize, 0, -1)

    outputLowerSeeds=[]
    for seed in lowerSeeds:
        if maskImage[seed[1],seed[0]]!=0: outputLowerSeeds.append(seed)

    return outputLowerSeeds

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dem", required=True, help="Path to Digital Elevation Map (DEM)")
    ap.add_argument("-s", "--size", required=True, help="Window size (squared)")
    ap.add_argument("-o", "--binOut", required=False, help="Path of the resulting binary image")
    ap.add_argument("-perc", "--percentile", required=False, help="The height percentile where most tops are expected to be")
    ap.add_argument("-mpt", "--minPixTop", required=True, help="Minimum pixels per top")
    ap.add_argument("-ref", "--refine", required=False, help="yes/no, do we refine the results by fusing nearby points?")
    ap.add_argument("-refRad", "--refineRadius", required=False, help="Radius for global refinement")
    ap.add_argument("-ts", "--topStep", required=True, help="Steps when choosing tree tops")
    ap.add_argument("-two", "--twoBands", required=False, help="Whether or not we do two bands")
    args = vars(ap.parse_args())

    #minNumPointsTree=2500
    #minNumPointsTreeLower=10000
    minNumPointsTree=25
    minNumPointsTreeLower=100

    minNumPointsTop=int(args["minPixTop"])
    minNumPointsTopLower=int(1.2*float(args["minPixTop"]))

    maxNumPointsTree=1950
    lowerPercent=1

    if (args["twoBands"] is not None) and (args["twoBands"]=="yes"):doLower=True
    else:doLower=False

    #print(args["dem"])
    dem2 = gdal.Open(args["dem"], gdal.GA_ReadOnly)

    dem=dem2.GetRasterBand(dem2.RasterCount).ReadAsArray().astype(float)
    del dem2

    dem[dem<0]=0 #eliminate non values

    maxDem=np.max(dem)

    #print(" shape of the DEM!!!!"+str(dem.shape))
    if dem is None:raise Exception("no DEM at "+str(args["dem"]))
    gray = dem
    #blurred=cv2.GaussianBlur(dem,(15,15),0)
    #blurred[dem==0]=0

    #cv2.imwrite("dem.jpg",(gray*(255/maxDem)).astype("uint8"))
    #cv2.imwrite("blurred.jpg",(gray*(255/maxDem)).astype("uint8"))
    #gray=dem

    if doLower:
        if args["percentile"] is not None:perc=int(args["percentile"])
        else: perc=70

        grayForPercentile=gray.copy()
        grayForPercentile[grayForPercentile==0]=np.nan
        cutOff=np.nanpercentile(grayForPercentile, perc)
        lowerCutOff=np.nanpercentile(grayForPercentile,lowerPercent)
        print("cutoff!!! "+str(cutOff)+" of "+str(maxDem)+"  at percentile "+str(perc)+" and lower at "+str(lowerCutOff))

        firstBand=ut.thresholdDEM(gray,cutOff,maxDem)
        secondBand=ut.thresholdDEM(gray,lowerCutOff,maxDem)
    else:
        firstBand=gray
        cutOff=0

    # loop over the sliding window for the first band of intensities
    seeds=[]
    lowerSeeds=[]
    #countLower=0
    for (x, y, window) in sliding_window(firstBand, stepSize=int(args["size"]), windowSize=(int(args["size"]), int(args["size"])),allImage=False):
        #print("window "+str(x)+" "+str(y))
        thisWindowSeeds=[]
        if np.sum(window>0)>minNumPointsTree:
            thisWindowSeeds=processWindow(window,args,minNumPointsTop,maxNumPointsTree,-1)

        if(len(thisWindowSeeds))>0:
            for localSeed in thisWindowSeeds:seeds.append((x+localSeed[1],y+localSeed[0]))
            #print("found seeds "+str(len(seeds)))

        if doLower:
            #now loop again with lower seeds
            lowerWindow=secondBand[y:y + int(args["size"]), x:x + int(args["size"])]
            thisWindowLowerSeeds=[]
            #if False:
            if np.sum(lowerWindow>0)>minNumPointsTreeLower:
                #print("                              going to process lower window ")
                thisWindowLowerSeeds=processWindow(lowerWindow,args,minNumPointsTopLower,maxNumPointsTree,cutOff)

            if(len(thisWindowLowerSeeds))>0:
                for localSeed in thisWindowLowerSeeds:lowerSeeds.append((x+localSeed[1],y+localSeed[0]))
                #print("found lower seeds "+str(len(lowerSeeds)))

    #print("going to refine Lower")
    if len(lowerSeeds)>0:refinedLower=refineLower(gray,seeds,lowerSeeds,args)
    else: refinedLower=[]
    print("Finished two bands with "+str(len(seeds))+" "+str(len(refinedLower)))

    highSeeds=list(seeds)
    outputImages(dem,highSeeds,refinedLower,args,cutOff)

if __name__ == "__main__":
    main()
