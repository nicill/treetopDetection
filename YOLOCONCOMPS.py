# import the necessary packages
import argparse

import cv2
import numpy as np
import math

import os
import sys

from imageUtils import sliding_window, binarizeWindow, refineTopDict,dictToTopsList

from sliding_window import eraseBorderPixels,paintTopsTrimNonCanopy,refineSeedsWithMaximums
import demUtils as ut
from collections import defaultdict

def boxesInWindow(x,y,s,boxList):
    """
    Given a starting point and size,
    find what boxes are in the window and return them
    in local window coordinates
    """
    retList = []
    #print((x,y,s))
    for x1,y1,x2,y2 in boxList:
        lx1 = float(x1) - x
        lx2 = float(x2) - x
        ly1 = float(y1) - y
        ly2 = float(y2) - y
        # ignoring boxes but by the window
        if 0 <= lx1 < s and 0 <= lx2 < s and 0 <= ly1 < s and 0 <= ly2 < s:
            #print(str((x1,y1,x2,y2))+" and "+str((lx1,ly1,lx2,ly2)))
            retList.append((lx1,ly1,lx2,ly2))
    return retList

def boxStats(im, box, percFilter, sillyC,otherC):
    """
       Given an image and a bounding box, find
       - max pixel in the box
       - Min pixel in top percentile of the box
       - count number of pixels in the center of the box
    """
    # gather box coordinates
    x1,y1,x2,y2 = [float(a) for a in box]

    # box center
    cx = int((x1 + x2)/2)
    cy = int((y1+ y2)/2)

    wMax = np.max(im)

    # now filter out values outside the box
    aux = im.copy()
    aux[:int(y1),:] = 0
    aux[int(y2):,:] = 0
    aux[:,:int(x1)] = 0
    aux[:,int(x2):] = 0
    #cv2.imwrite("./debug/comp"+str(sillyC)+"F"+str(otherC)+".png",aux*(255/wMax))

    # find max
    boxMax = np.max(aux)

    # find min
    aux2 = aux.copy()
    aux2[aux2==0] = np.nan
    boxPerc = np.nanpercentile(aux2,percFilter)

    # count points
    aux[aux<boxPerc] = 0
    topPoints = np.sum(aux>0)

    #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(aux)
    #return boxMax,boxPerc,topPoints,(maxLoc[1],maxLoc[0])

    # returning centers of the bounding box and not maximums
    # This is done because trying to return the maximum was quite a lot worse
    return boxMax,boxPerc,topPoints,(cy,cx)

def findTopsYCC(comp,args,minPix,verbose = False): #receive a grayscale image of a connected component, threshold it repeatedly until you find all tree tops
    pixMinConComp=minPix

    nonBlack=np.sum(comp!=0)

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
        labelDict=defaultdict(lambda:[])
        for top in tops:
            labelDict[labelImage[top[0],top[1]]].append( (comp[top[0],top[1]],top)  )
        if verbose: print("Label dictionary "+str(labelDict))

        # for each component already in the dict,
        # add the top point forcefully if necessary
#        for label,tupList in labelDict.items():
            # also update the maximum if necessary,
            # and if this happens this is bad
 #           listOfHeights = [x for x,y in tupList]

 #           auxWin=comp.copy()
 #           auxWin[labelImage!=label]=0

 #           (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(auxWin)

 #           if maxVal>max(listOfHeights): labelDict[label].append( (comp[top[0],top[1]],top))


        # Now call function to clean up inside connected components, keep higher within those at distance 2 epsilon
        labelDict = refineTopDict(labelDict,epsilon)
        if verbose: print("Label dictionary after refinement "+str(labelDict))

        tops = dictToTopsList(labelDict)


        # traverse all current labels (but the background), the highest point of ALL get added as candidate tree tops
        candidateTops=[]
        for l in range(1,numLabels):
            currentCount=np.count_nonzero(labelImage==l)
            if currentCount>pixMinConComp and not l in labelDict:
                # find the highest point in this connected component
                #auxWin = comp.copy()
                #auxWin[labelImage != l ] = 0

                #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(auxWin)
                #candidateTops.append((int(maxLoc[1]),int(maxLoc[0])))
                #if verbose: print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^appending top with altitude "+str(maxVal))
                
                #add instead the center of the connected component.
                candidateTops.append((int(centroids[l][1]),int(centroids[l][0])))
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

def processWindowComp(win,args, boxLocal,sillyC):

    # Setup window parameters
    steps = 150
    cutPerc = 75

    wMax = np.max(win)
    win2 = win.copy()
    otherCount = 0
    statsBoxes = []
    for box in boxLocal:
        #print(box)
        x1,y1,x2,y2 = [float(a) for a in box]
        cv2.rectangle(win2,(int(x1),int(y1)),(int(x2),int(y2)),125,1)
        statsBoxes.append(boxStats(win, box, cutPerc, sillyC,otherCount))
        #bM,b75,bP = boxStats(win, box, sillyC,otherCount)
        #print("box max "+str(bM)+" 75 "+str(b75)+" points "+str(bP))
        otherCount+=1

    avMax = sum(el[0] for el in statsBoxes)/len(statsBoxes)
    minTOPS = min(el[1] for el in statsBoxes)
    avP = sum(el[2] for el in statsBoxes)/len(statsBoxes)
    minP = min(el[2] for el in statsBoxes)
    centers = [el[3] for el in statsBoxes]
    #print("component averages/min max, perc, points "+str(avMax)+"  "+str(minTH)+"  "+str(avP)+" "+str(minP)+" "+str(centers))

    argsLocal = args.copy()
    argsLocal["topStep"] = max((avMax-minTOPS)/steps,0.05)
    argsLocal["refineRadius"] = max(int(minP/10),5)
    argsLocal["minPixTop"] = max(minP,30)

    #argsLocal["topStep"] = 0.05
    #argsLocal["refineRadius"] = 5
    #argsLocal["minPixTop"] = 30
    argsLocal["thpercentile"] = 25
    argsLocal["eroKernS"] = 5
    args["eroIt"] = 1

    print(argsLocal)

    # Erase by the min of the top boxes
    #aux = win2.copy()
    #aux[aux==0] = np.nan
    #minFloor = np.nanpercentile(aux,1)
    #perc = minTOPS - (minTOPS - minFloor )/20
    #win2[win2 < perc] = 0
    #print("erasing under "+str(perc))
    #sys.exit()

    global stupidCount
    stupidCount = 0
    binarized,stupidCount = binarizeWindow(win,stupidCount,lowerPerc = int(argsLocal["thpercentile"]), eroKernS = int(argsLocal["eroKernS"]), eroIt = int(args["eroIt"]), debugImages = argsLocal["debugImages"])
    #win2[binarized == 0] = 0
    #cv2.imwrite("./debug/compBoxes"+str(sillyC)+"BEFORE.png",win*(255/wMax))
    #cv2.imwrite("./debug/compBoxes"+str(sillyC)+".png",win2*(255/wMax))

    #binarized = np.zeros((win2.shape[0],win2.shape[1]),np.uint8)
    #binarized[win2>0] = 255 # here we could be using binarize window from imageutils

    # now, for every connected component, find tops
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(binarized)

    # here we start with the YOLO Seeds
    seeds = centers
    print("processWindow:: Processing Window, found number of con comp: "+str(numLabels))

    # find tree tops only in connected components that may contain a tree
    for l in range(1,numLabels):
        #for each label, count tops
        if stats[l,cv2.CC_STAT_AREA] > argsLocal["minPixTop"]:
            #print("                                  processWindow:: window of size : "+str(stats[l,cv2.CC_STAT_AREA]))
            thisComponent=win.copy()
            thisComponent[labelImage != l] = 0

            #recomp = thisComponent.copy()
            #recomp[thisComponent>0]=255

            thisCompTops=findTopsYCC(thisComponent,argsLocal,argsLocal["minPixTop"])
            #print("found "+str(len(thisCompTops)))
            seeds.extend(thisCompTops)

            #for seed in seeds:
            #    cv2.circle(recomp, seed, 1, (125,125, 125), -1)
            #recomp = cv2.resize(recomp, (recomp.shape[1]*10, recomp.shape[0]*10), interpolation = cv2.INTER_LINEAR)
            #cv2.imwrite("./debug/"+str(stupidCount)+"comp"+str(l)+".jpg",recomp)


    #sys.exit()
    # maybe refine seeds in the window, if two seed are too close, eliminate one of them (or something like that)
    return seeds


def outputYC(dem,seeds,args,cutoff=0,index=None):
    if False:
        print("not refining")
        if args["binOut"] is not None: out_binary = args["binOut"]
        else: out_binary = filename + "_binaryMethodCCHEIGHT"+ file_extension

        circleSize=2
        maskImage=255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)
        for seed in seeds:
            cv2.circle(dem, seed, circleSize, (255, 255, 255), -1)
            # also create binary result image
            cv2.circle(maskImage, seed, circleSize, 0, -1)
        cv2.imwrite(out_binary, maskImage)
        print("not refining, exitting")
        return

    if len(seeds)<2: return 255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)

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

    #cv2.imwrite("ruumip.jpg", dem)
    cv2.imwrite(out_binary, maskImage)

# examples of execution python YOLOCONCOMPS.py -d ./KOIdataHR/ROI_DEM2.tif -box ./combination/resampledCoords.txt -s 500 -o rumpup.png -refRad 7; python crownSegmenterEvaluator.py rumpup.png KOIdataHR/Label_image.tif
# python YOLOCONCOMPS.py -d ./KOIdataHR/DEMNOBCKGD.tif -box ./combination/boxesDec24.txt -s 500 -o rumpup.png -refRad 7; python crownSegmenterEvaluator.py rumpup.png KOIdataHR/Label_image.tif
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dem", required=True, help="Path to Digital Elevation Map (DEM)")
    ap.add_argument("-box", "--boxFile", required=True, help="Path to File with boxes from YOLO")
    ap.add_argument("-s", "--size", required=True, help="Window size (squared)")
    ap.add_argument("-o", "--binOut", required=False, help="Path of the resulting binary image")
    #ap.add_argument("-mpt", "--minPixTop", required=True, help="Minimum pixels per top")
    ap.add_argument("-refRad", "--refineRadius", required=True, help="Radius for global refinement")
    #ap.add_argument("-ref", "--refine", required=False, help="yes/no, do we refine the results by fusing nearby points?")
    ap.add_argument("-imDebug", "--debugImages", required=False, help="Name of the directory to store connected componen images")
    args = vars(ap.parse_args())

    border = 15
    if args["debugImages"] is None: args["debugImages"] = "NO"
    args["refine"] = "no"

    print(args["dem"])
    dem = cv2.imread(args["dem"],cv2.IMREAD_UNCHANGED)
    if dem is None:
        print(str(args["dem"])+ "not found ")
        sys.exit(0)

    print(args["boxFile"])
    with open(args["boxFile"]) as f:
        boxes = [ tuple(line.strip().split(" ")) for line in f.readlines()]
    print(str(len(boxes))+" boxes read ")
    #print(boxes)


    # The DEM has a thin layer of wrong values in the outer part
    # filter them out
    filterOuterPixels = False
    if filterOuterPixels:
        eraseBorderPixels(dem,border)

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

    maxDem=np.max(dem)
    print("Maximum value for this DEM (after subtracting the minimum) was "+str(maxDem))
    
    # maximum outlier removal (does not seem to help)
    #almostMaxDem = np.nanpercentile(dem,99)
    #print("dem 99 perc "+str(almostMaxDem))
    #dem[dem>almostMaxDem] = 0
    #maxDem=np.max(dem)
    #print("Maximum value for this DEM (after max outlier removal) was "+str(maxDem))

    cv2.imwrite("demOU.jpg",(gray*(255/maxDem)).astype("uint8"))

    # loop over the sliding window
    seeds=[]
    windowOverlap = 0.2
    sillyC = 0
    for (x, y, window) in sliding_window(gray, stepSize=int(int(args["size"])*(1-windowOverlap)), windowSize=(int(args["size"]), int(args["size"])),allImage=False):
        #print("window "+str(x)+" "+str(y))
        biW = boxesInWindow(x,y,int(args["size"]),boxes)
        if len(biW) > 0:
            wMax = np.max(window)
            #cv2.imwrite("./debug/comp"+str(sillyC)+".png",window*(255/wMax))

            thisWindowSeeds = processWindowComp(window,args,biW,sillyC)


            if(len(thisWindowSeeds))>0:
                for localSeed in thisWindowSeeds:seeds.append((x+localSeed[1],y+localSeed[0]))
                #print("found seeds "+str(len(seeds)))
        sillyC+=1

    print("Finished with "+str(len(seeds)))

    outputYC(dem,seeds,args)

if __name__ == "__main__":
    main()
