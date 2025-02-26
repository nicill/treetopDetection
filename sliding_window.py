# import the necessary packages
import argparse

import cv2
import numpy as np
import math

import os
import sys

# for the function to turn binary files into lists of points
import demUtils as ut
from imageUtils import (sliding_window, binarizeWindow,refineTopDict,
                        dictToTopsList,eraseBorderPixels)
from pathlib import Path

stupidCount = 0


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
def findTops(comp,args,minPix,verbose = False): #receive a grayscale image of a connected component, threshold it repeatedly until you find all tree tops
    #pixMinConComp=int(args["minPixTop"])
    pixMinConComp=minPix

    nonBlack=np.sum(comp!=0)

    #print("gradpixperc "+str(gradientPixelPerc))
    if verbose: print("This image should contains between "+str(minTrees)+" and "+str(maxTrees)+" Trees ")
    #if nonBlack<minPix:return []

    # retieve epsilon
    epsilon = int(args["refineRadius"])

    #loop over different bands of the DEM, from top to bottom
    demIstep=float(args["topStep"])
    demIstart=np.max(comp)-demIstep
    demIend=np.min(comp[np.nonzero(comp)])
    demLength=demIstart-demIend
    demNumSteps=demLength/demIstep

    maxIt=int(1*demNumSteps)

    numIterations=1
    finished=False
    #numberTopsFoundList=[]
    tops=[]
    aupo=0
    while demIstart+demIstep>demIend:
        # First get the upper band of the DEM
        aupo+=1
        if verbose: print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::findtops between "+str(demIstart)+" and "+str(demIstart+demIstep*numIterations))
        thisBand=ut.thresholdDEMBinarised(comp,demIstart)

        # compute connected components here
        numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(thisBand)
        #print("this band has this many connected components "+str(numLabels))

        # for every existing top, make a note of its label in this class in a dictionary
        labelDict={}
        for top in tops:
            thisTopLabel = labelImage[top[0],top[1]]
            if thisTopLabel not in labelDict: labelDict[thisTopLabel]=[(comp[top[0],top[1]],top)]
            else: labelDict[thisTopLabel].append( (comp[top[0],top[1]],top)  )
        if verbose: print("Label dictionary "+str(labelDict))

        # for each component already in the dict,
        # add the top point forcefully if necessary
        for label,tupList in labelDict.items():
            # also update the maximum if necessary,
            # and if this happens this is bad
            listOfHeights = [x for x,y in tupList]

            auxWin=comp.copy()
            auxWin[labelImage!=label]=0

            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(auxWin)

            if maxVal>max(listOfHeights): labelDict[label].append( (comp[top[0],top[1]],top))


        # Now call function to clean up inside connected components, keep higher within those at distance 2 epsilon
        labelDict = refineTopDict(labelDict,epsilon)
        if verbose: print("Label dictionary after refinement "+str(labelDict))

        tops = dictToTopsList(labelDict)


        # traverse all current labels (but the background), the highest point of ALL get added as candidate tree tops
        candidateTops=[]
        for l in range(1,numLabels):
            currentCount=np.count_nonzero(labelImage==l)
            if currentCount>pixMinConComp and not l in labelDict:
            #if currentCount>pixMinConComp :
                # find the highest point in this connected component
                auxWin = comp.copy()
                auxWin[labelImage != l ] = 0

                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(auxWin)
                candidateTops.append((int(maxLoc[1]),int(maxLoc[0])))
                if verbose: print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^appending top with altitude "+str(maxVal))
            elif verbose :
                if l in labelDict : print("existing label "+str(l))
                else: print("small top "+str(currentCount))

        tops.extend(candidateTops)

        #print("now have "+str(len(tops))+" treeTops ")
        demIstart-=demIstep
        numIterations+=1
        #numberTopsFoundList.append(len(tops))

        if numIterations>=maxIt :
            #print("Ofinished tops in "+str(numIterations)+", finding "+str(len(tops))+" "+str(numberTopsFoundList)+"\n\n\n")
            return tops


    #if  len(tops)>maxTrees or len(tops)<minTrees:
    #    print("WRONG NUMBER!!!!!!!!!!!!!! was:"+str(len(tops))+" SHOULD HAVE BEEN between "+str(minTrees)+" and "+str(maxTrees)+" Trees ")

        if args["debugImages"] != "NO":
            Path(args["debugImages"]).mkdir(parents=True, exist_ok=True)

            #show band and found tops
            # visualization purposes only
            comp2 = thisBand.copy()
            for x,y in tops:
                cv2.circle(comp2, (y,x), 2, 155, -1)
            comp2 = cv2.resize(comp2, (comp2.shape[1]*10, comp2.shape[0]*10), interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(args["debugImages"]+"/COMPONENT"+str(stupidCount-1)+"IS"+str(aupo)+"comp.png",comp2)

    return tops

def processWindow(win,args,minNumPointsTree,maxNumPointsTree):
    # binarize image, erode it a little, compute connected connectedComponents
    global stupidCount
    binarized,stupidCount = binarizeWindow(win,stupidCount,lowerPerc = int(args["thpercentile"]), eroKernS = int(args["eroKernS"]), eroIt = int(args["eroIt"]), debugImages = args["debugImages"])

    # now, for every connected component, find tops
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(binarized)

    seeds=[]

    #cv2.imwrite("./out/"+str(stupidCount)+"ou.jpg",(255/max(np.unique(labelImage)))*labelImage)

    #print("processWindow:: Processing Window, found number of con comp: "+str(numLabels))
    #print(minNumPointsTree)

    # find tree tops only in connected components that may contain a tree
    for l in range(1,numLabels):
        #for each label, count tops
        if stats[l,cv2.CC_STAT_AREA] > minNumPointsTree:
            #print("                                  processWindow:: window of size : "+str(stats[l,cv2.CC_STAT_AREA]))
            thisComponent=win.copy()
            thisComponent[labelImage != l] = 0

            #recomp = thisComponent.copy()
            #recomp[thisComponent>0]=255
            #recomp = cv2.resize(recomp, (recomp.shape[1]*10, recomp.shape[0]*10), interpolation = cv2.INTER_LINEAR)
            #cv2.imwrite("./out/"+str(stupidCount)+"comp"+str(l)+".jpg",recomp)
            thisCompTops=findTops(thisComponent,args,minNumPointsTree)
            seeds.extend(thisCompTops)

    # maybe refine seeds in the window, if two seed are too close, eliminate one of them (or something like that)
    return seeds

def paintTopsTrimNonCanopy(dem,seeds,circleSize,cutoff,eroK = 5, eroIt =1):
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
        #print(thisTopHeight )
        #print(higherTop-thisTopHeight)
        #print(step)
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
            #print("checking "+str(thisTop)+" "+str(thisTopHeight))
            for thisTop,thisTopHeight in x:
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

def outputImages(dem,seeds,args,cutoff=0,index=None):

    if len(seeds)<2: return 255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)

    #If refine, paint the high seeds first and carefully paint the others after
    if (args["refine"] is not None) and (args["refine"]=="yes") and (args["refineRadius"] is not None):

        print("refining")
        maskImage=255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)
        circleSize=int(args["refineRadius"])
        #print("CIRCLESIZE "+str(circleSize))

        #prepare for refinement, disconnect floor and lower regions
        maskImageLow = paintTopsTrimNonCanopy(dem,seeds,circleSize,cutoff)

        #join the two masks
        maskImage[maskImageLow==0]=0

        #refine by taking only the highest point in every resulting region and eliminating small regions
        outputSeeds=refineSeedsWithMaximums(dem,maskImage,int(args["refineRadius"]),seeds)
        seeds = outputSeeds
    else:
        print("Skipping seed refinement!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #print("number of refined seeds! "+str(len(outputSeeds)))

    # Once the method is finished, paint the detected points
    # paint them as circles over the DEM and the image
    #print(" Total seeds, found "+str(len(seeds)))

    print("going to write the following number of seeds "+str(len(seeds)))
    circleSize=2
    maskImage=255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)
    for seed in seeds:
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
    #print("out binary will be "+out_binary)

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

def morphologyOperations(image, kernel_size, min_height,iter=1):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    ret,thresh = cv2.threshold(image,min_height,255,cv2.THRESH_TOZERO)
    im_erode = cv2.erode(thresh,kernel,iterations=iter)

    blurred = cv2.GaussianBlur(im_erode,(kernel_size,kernel_size),0)

    im_dilate = cv2.dilate(blurred,kernel,iterations=iter)
    im_comp = cv2.compare(blurred,im_dilate,cv2.CMP_GE)

    im_erode = cv2.erode(blurred,kernel,iterations=1)
    im_plateu = cv2.compare(blurred,im_erode,cv2.CMP_GT)
    im_and = cv2.bitwise_and(im_comp,im_plateu)

    return im_and

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dem", required=True, help="Path to Digital Elevation Map (DEM)")
    ap.add_argument("-s", "--size", required=True, help="Window size (squared)")
    ap.add_argument("-o", "--binOut", required=False, help="Path of the resulting binary image")
    ap.add_argument("-thPerc", "--thpercentile", required=True, help="The height percentile at which local windows are thresholded")
    ap.add_argument("-mpt", "--minPixTop", required=True, help="Minimum pixels per top")
    ap.add_argument("-refRad", "--refineRadius", required=True, help="Radius for global refinement")
    ap.add_argument("-ref", "--refine", required=False, help="yes/no, do we refine the results by fusing nearby points?")
    ap.add_argument("-ts", "--topStep", required=True, help="Steps when choosing tree tops")
    ap.add_argument("-eS", "--eroKernS", required=True, help="Size of the erosion kernel used to take out the floor part")
    ap.add_argument("-eIt", "--eroIt", required=True, help="Number of iterations of the erosion kernel used to take out the floor part")
    ap.add_argument("-imDebug", "--debugImages", required=False, help="Name of the directory to store connected componen images")
    args = vars(ap.parse_args())

    minNumPointsTree=400
    minNumPointsTreeLower=100

    minNumPointsTop=int(args["minPixTop"])
    minNumPointsTopLower=int(1.2*float(args["minPixTop"]))

    if args["debugImages"] is None: args["debugImages"] = "NO"

    maxNumPointsTree=1950
    lowerPercent=1

    print(args["dem"])
    dem = cv2.imread(args["dem"],cv2.IMREAD_UNCHANGED)
    if dem is None:
        print(str(args["dem"])+ "not found ")
        exit(0)

    # The DEM has a thin layer of wrong values in the outer part
    # filter them out
    filterOuterPixels = True
    if filterOuterPixels:
        eraseBorderPixels(dem)

    #Filter non values and outliers
    dem[ dem<0 ] = 0 #eliminate non values

    # take out the min (sort of)
    demPerc=dem.copy()
    demPerc[demPerc==0] = np.nan
    minDem = np.nanpercentile(demPerc,1)

    print("Minimum value for this DEM was  "+str(minDem))

    dem = dem - minDem
    dem[dem<0] = 0
    gray = dem

    #Apply morphology
    #sizeMorph = 5
    #iterMorph = 1
    #dem2 = morphologyOperations(dem.copy(), sizeMorph, 0, iter = iterMorph)
    #cv2.imwrite("dem2.jpg",dem2)

    maxDem=np.max(dem)
    print("Maximum value for this DEM (after subtracting the minimum) was "+str(maxDem))

    cv2.imwrite("dem.jpg",(gray*(255/maxDem)).astype("uint8"))


    firstBand=gray
    cutOff=0

    # loop over the sliding window for the first band of intensities
    seeds=[]
    windowOverlap = 0.2
    for (x, y, window) in sliding_window(firstBand, stepSize=int(int(args["size"])*(1-windowOverlap)), windowSize=(int(args["size"]), int(args["size"])),allImage=False):
        #print("window "+str(x)+" "+str(y))
        thisWindowSeeds=[]
        if np.sum(window>0)>minNumPointsTree:
            thisWindowSeeds = processWindow(window,args,minNumPointsTop,maxNumPointsTree)

        if(len(thisWindowSeeds))>0:
            for localSeed in thisWindowSeeds:seeds.append((x+localSeed[1],y+localSeed[0]))
            #print("found seeds "+str(len(seeds)))

    print("Finished with "+str(len(seeds)))

    outputImages(dem,seeds,args,cutOff)

if __name__ == "__main__":
    main()
