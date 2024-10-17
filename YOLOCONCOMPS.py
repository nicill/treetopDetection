# import the necessary packages
import argparse

import cv2
import numpy as np
import math

import os
import sys

from imageUtils import sliding_window

from sliding_window import eraseBorderPixels,findTops,outputImages

def boxesInWindow(x,y,s,boxList):
    """
    Given a starting point and size,
    find what boxes are in the window and return them
    in local window coordinates
    """
    retList = []
    print((x,y,s))
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

def boxStats(im, box, sillyC,otherC):
    """
       Given an image and a bounding box, find
       - max pixel in the box
       - Min pixel in a small center of the box (25%?)
       - count number of pixels in the center of the box
    """
    # gather box coordinates
    x1,y1,x2,y2 = [float(a) for a in box]

    wMax = np.max(im)

    # now filter out values outside the box
    aux = im.copy()
    aux[:int(y1),:] = 0
    aux[int(y2):,:] = 0
    aux[:,:int(x1)] = 0
    aux[:,int(x2):] = 0

    #cv2.imwrite("./debug/comp"+str(sillyC)+"F"+str(otherC)+".png",aux*(255/wMax))
    percFilter = 60

    # find max
    boxMax = np.max(aux)

    # find min
    aux2 = aux.copy()
    aux2[aux2==0] = np.nan
    box75 = np.nanpercentile(aux2,percFilter)

    # count points
    aux[aux<box75] = 0

    topPoints = np.sum(aux>0)

    return boxMax,box75,topPoints

def processWindowComp(win,args, boxLocal,sillyC):
    wMax = np.max(win)

    win2 = win.copy()
    otherCount = 0
    statsBoxes = []
    for box in boxLocal:
        #print(box)
        x1,y1,x2,y2 = [float(a) for a in box]
        cv2.rectangle(win2,(int(x1),int(y1)),(int(x2),int(y2)),125,1)
        statsBoxes.append(boxStats(win, box, sillyC,otherCount))
        #bM,b75,bP = boxStats(win, box, sillyC,otherCount)
        #print("box max "+str(bM)+" 75 "+str(b75)+" points "+str(bP))
        otherCount+=1
    avMax = sum(el[0] for el in statsBoxes)/len(statsBoxes)
    minTH = min(el[1] for el in statsBoxes)
    avP = sum(el[2] for el in statsBoxes)/len(statsBoxes)
    minP = min(el[2] for el in statsBoxes)
    #print("component averages/min max, 75, points "+str(avMax)+"  "+str(minTH)+"  "+str(avP))

    win2[win2<minTH] = 0
    #cv2.imwrite("./debug/compBoxes"+str(sillyC)+".png",win2*(255/wMax))

    # Setup window parameters
    steps = 25
    args["topStep"] = (avMax-minTH)/steps
    #args["refineRadius"] = int(minP/10)
    args["refineRadius"] = 3
    args["minPixTop"] = avP/20


    binarized = np.zeros((win2.shape[0],win2.shape[1]),np.uint8)
    binarized[win2>0] = 255 # here we could be using binarize window from imageutils

    # now, for every connected component, find tops
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(binarized)

    seeds=[]

    #print("processWindow:: Processing Window, found number of con comp: "+str(numLabels))
    #print(minNumPointsTree)

    # find tree tops only in connected components that may contain a tree
    for l in range(1,numLabels):
        #for each label, count tops
        if stats[l,cv2.CC_STAT_AREA] > args["minPixTop"]:
            #print("                                  processWindow:: window of size : "+str(stats[l,cv2.CC_STAT_AREA]))
            thisComponent=win.copy()
            thisComponent[labelImage != l] = 0

            #recomp = thisComponent.copy()
            #recomp[thisComponent>0]=255
            #recomp = cv2.resize(recomp, (recomp.shape[1]*10, recomp.shape[0]*10), interpolation = cv2.INTER_LINEAR)
            #cv2.imwrite("./out/"+str(stupidCount)+"comp"+str(l)+".jpg",recomp)
            thisCompTops=findTops(thisComponent,args,args["minPixTop"])
            print("found "+str(len(thisCompTops)))
            seeds.extend(thisCompTops)

    # maybe refine seeds in the window, if two seed are too close, eliminate one of them (or something like that)
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dem", required=True, help="Path to Digital Elevation Map (DEM)")
    ap.add_argument("-box", "--boxFile", required=True, help="Path to File with boxes from YOLO")
    ap.add_argument("-s", "--size", required=True, help="Window size (squared)")
    ap.add_argument("-o", "--binOut", required=False, help="Path of the resulting binary image")
    #ap.add_argument("-mpt", "--minPixTop", required=True, help="Minimum pixels per top")
    #ap.add_argument("-refRad", "--refineRadius", required=True, help="Radius for global refinement")
    #ap.add_argument("-ref", "--refine", required=False, help="yes/no, do we refine the results by fusing nearby points?")
    ap.add_argument("-imDebug", "--debugImages", required=False, help="Name of the directory to store connected componen images")
    args = vars(ap.parse_args())

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

    maxDem=np.max(dem)
    print("Maximum value for this DEM (after subtracting the minimum) was "+str(maxDem))

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
                print("found seeds "+str(len(seeds)))
        sillyC+=1

    print("Finished with "+str(len(seeds)))

    outputImages(dem,seeds,args)

if __name__ == "__main__":
    main()
