import numpy as np
import cv2
import sys
import crownSegmenterEvaluator as CSE

from scipy.spatial.distance import directed_hausdorff
from imageUtils import sliding_window,distPoints, binarizeWindowReturnDEM, storePrettyDEM

from sklearn.neighbors import KDTree

def avMinDistCloserTop(list1):
    if len(list1)==0: return -1

    newList1=np.asarray([[x,y] for x,y in list1 ])
    newList2=np.asarray([[x,y] for x,y in list1 ])

    kdt = KDTree(newList2, leaf_size=30, metric='euclidean')
    dist,ind=kdt.query(newList1, k=2)
    sum=0
    for d in dist:
        if len(d)>0:
            sum+=d[1]

    return sum/len(list1)

def countLostTops(dem, treeTops):

    mask = cv2.threshold(255-treeTops, 80, 255, cv2.THRESH_BINARY)[1]
    #compute connected components
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

    count=0
    for i in range(1,len(centroids)):
        x=centroids[i][1]
        y=centroids[i][0]

        #print(x)

        if dem[int(x)][int(y)]==0:count+=1

    return 100*count/len(centroids)

#https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
def countLostTopsRegionSize(dem, treeTops,minSize=10):

    mask = cv2.threshold(255-treeTops, 80, 255, cv2.THRESH_BINARY)[1]
    #compute connected components
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

    dem[dem>0]=255
    numLabels2, labelImage2,stats2, centroids2 = cv2.connectedComponentsWithStats(dem)

    cv2.imwrite("thresholdedDEM2.jpg",dem)


    print("tops "+str(numLabels))
    print("arbres "+str(numLabels2))

    count=0
    for i in range(1,len(centroids)):
        x=int(centroids[i][1])
        y=int(centroids[i][0])

        #print(x)
        #if dem[x][y] == 0 : print("zero "+str(centroids[i]))

        currentLabel=labelImage2[int(x)][int(y)]
        if currentLabel==0:
            #print("found black top at "+str(centroids[i]))
            count+=1
        else:
            pointsThisLabel=stats2[currentLabel,cv2.CC_STAT_AREA]
            if pointsThisLabel<minSize:
                print(str(centroids[i])+" had "+str(pointsThisLabel))
                count+=1
#            else:
#                print(str(centroids[i])+" was ok with "+str(pointsThisLabel))

    return 100*count/len(centroids)


#https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
def averageTopRegionSizePercentage(dem,treeTops,th):

    #threshold DEM
    dem[dem<th]=0

    mask = cv2.threshold(255-treeTops, 80, 255, cv2.THRESH_BINARY)[1]
    #compute connected components
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

    #binarize dem
    demAux=dem.copy()
    demAux[dem>0.1]=255
    demAux[dem<=0.1]=0
    numLabels2, labelImage2,stats2, centroids2 = cv2.connectedComponentsWithStats(demAux.astype("uint8"))

    dictTopInfo={}

    count=0
    for i in range(1,len(centroids)):
        x=centroids[i][1]
        y=centroids[i][0]

        currentLabel=labelImage2[int(x)][int(y)]
        #print("iteration "+str(i)+" current label "+str(currentLabel))

        if currentLabel not in dictTopInfo:
            if currentLabel==0:currentArea=-1
            else:currentArea=stats2[currentLabel,cv2.CC_STAT_AREA]
            dictTopInfo[currentLabel]=(1,currentArea)
        else:
            dictTopInfo[currentLabel]=(dictTopInfo[currentLabel][0]+1,dictTopInfo[currentLabel][1])

    return dictTopInfo

def thresholdDEMStats(dem,treeTops,th):

    #threshold DEM
    demAux=dem.copy()
    demAux[dem<th]=0

    mask = cv2.threshold(255-treeTops, 80, 255, cv2.THRESH_BINARY)[1]
    #compute connected components
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

    count=0
    for i in range(1,len(centroids)):
        x=centroids[i][1]
        y=centroids[i][0]

        #print(x)

        if demAux[int(x)][int(y)]==0:count+=1

    #cv2.imwrite("thresholdedDEM2.jpg",demAux)
    return 100-(100*count/len(centroids)),100*np.sum(demAux!=0)/np.sum(dem!=0)

def thresholdDEM(dem,th1,th2):
    #print("thresholding at "+str(th1)+" "+str(th2))
    #threshold DEM
    demAux=dem.copy()
    demAux[dem<th1]=0
    demAux[dem>th2]=0
    #print(np.sum(demAux>0))
    return demAux

def thresholdDEMBinarised(dem,th1,th2 = None):
    #print("thresholding at "+str(th1)+" "+str(th2))
    #threshold DEM
    demAux=dem.copy()

    demAux[dem<th1]=0
    if th2 is not None: demAux[dem>th2]=0
    demAux[demAux!=0]=255
    #print("                bigger than zero "+str(np.sum(demAux>0)))

    return demAux.astype("uint8")

def isInLayer(center,layer):
    return layer[int(center[0]),int(center[1])]==255

def labelIMResample(file):
    """
    resample labelImage
    """
    lIM = cv2.imread(file,cv2.IMREAD_UNCHANGED)
    if lIM is None: raise Exception("shit")

    lIMW = lIM.astype("uint16").copy()
    lIMW[lIMW > 0] = 255 #make binary image

    resizedW = cv2.resize(lIMW,(1491,5672),cv2.INTER_NEAREST).astype("uint8")
    resizedLabels = cv2.resize(lIM.astype("uint16"),(1491,5672),cv2.INTER_NEAREST).astype("uint16")

    # to ward of wrong values, eliminate possible border pixels
    resizedLabels[resizedW<255]=0
    cv2.imwrite("newLabels.tif",resizedLabels)


def main(argv):

    #argv[1] contains mode    cv2.imwrite("hkl.jpg",demAux)

    if (int(argv[1]))==0: # count how many tree tops are in the floor, argv[1] contains the DEM and argv[2] the treetops

        dem = cv2.imread(argv[2],0)
        if dem is None:raise Exception("demUtils no DEM at "+str(argv[2]))

        treeTops = cv2.imread(argv[3],0)
        if treeTops is None:raise Exception("demUtils no treeTops at "+str(argv[3]))

        print("tree tops lost in dem "+str(countLostTops(dem,treeTops)))
    elif (int(argv[1]))==1: # paint the treeTops bigger

        treeTops = cv2.imread(argv[2],0)
        if treeTops is None:raise Exception("demUtils no treeTops at "+str(argv[2]))

        maskImage=treeTops.copy()
        mask = cv2.threshold(255-treeTops, 10, 255, cv2.THRESH_BINARY)[1]
        #compute connected components
        numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

        circleSize=15
        for seed in centroids[1:]:
            cv2.circle(maskImage, (int(seed[0]),int(seed[1])), circleSize, 20, -1)

        cv2.imwrite(argv[3],maskImage)

    elif (int(argv[1]))==2: # threshold the DEM and see how many tops remain

        #dem = cv2.imread(argv[2],0)
        #if dem is None:raise Exception("demUtils no DEM at "+str(argv[2]))

        dem2 = gdal.Open(argv[2], gdal.GA_ReadOnly)
        for x in range(1, dem2.RasterCount + 1):
            band = dem2.GetRasterBand(x)
            #cv2.imwrite("./band"+str(x)+".png",band.ReadAsArray())
            array = band.ReadAsArray().astype(np.float)
            #print(array)

        dem=dem2.GetRasterBand(dem2.RasterCount).ReadAsArray().astype(np.float)

        dem[dem<0]=0 #eliminate non values
        maxDem=np.max(dem)
        gray = cv2.GaussianBlur(dem,(5,5),0)


        treeTops = cv2.imread(argv[3],0)
        if treeTops is None:raise Exception("demUtils no treeTops at "+str(argv[3]))

        grayForPercentile=gray.copy()
        grayForPercentile[grayForPercentile==0]=np.nan

        thresholdList=[]
        for i in range(10,100,10):
            thresholdList.append(np.nanpercentile(grayForPercentile, i))

        i=0
        while i<len(thresholdList):
            erasedTops,erasedPixPerc=thresholdDEMStats(dem,treeTops,thresholdList[i])
            #print("tree tops remaining at threshold "+str(th)+" "+"{:.2f}".format(erasedTops)+" with remaining pixel perc "+"{:.2f}".format(erasedPixPerc),flush=True)
            print("Percentile: "+str(10*(i+1))+" : "+"{:.2f}".format(thresholdList[i])+" "+"{:.2f}".format(erasedTops)+" "+"{:.2f}".format(erasedPixPerc),flush=True)
            i+=1

    elif (int(argv[1]))==3: # count how many tree tops are in the floor, argv[1] contains the DEM and argv[2] the treetops

        dem = cv2.imread(argv[2],0)
        if dem is None:raise Exception("demUtils no DEM at "+str(argv[2]))

        treeTops = cv2.imread(argv[3],0)
        if treeTops is None:raise Exception("demUtils no treeTops at "+str(argv[3]))

        print("tree tops lost in dem "+str(countLostTopsRegionSize(dem,treeTops)))
    elif (int(argv[1]))==4: # average of the distance to the closest tree top
        laist=CSE.listFromBinary(argv[2])
        print("!!")

        print(avMinDistCloserTop(laist))
    elif (int(argv[1]))==5: # take a list of tops and a mask and output the tops in the mask
        topList=CSE.listFromBinary(argv[2])
        mask=255-cv2.imread(argv[3],cv2.IMREAD_GRAYSCALE)
        if mask is None:raise Exception("demutils5 mask not read correclty")

        outputFileName=argv[4]
        maskImage=255*np.ones((mask.shape[0],mask.shape[1],1),dtype=np.uint8)
        circleSize=10

        for cent in topList:
            if isInLayer((cent[1],cent[0]),mask):cv2.circle(maskImage, (int(cent[0]),int(cent[1])), circleSize, 0, -1)
        cv2.imwrite(outputFileName,maskImage)
    elif (int(argv[1]))==6: # threshold the DEM and compute the average region size for the tops

        dem2 = gdal.Open(argv[2], gdal.GA_ReadOnly)
        for x in range(1, dem2.RasterCount + 1):
            band = dem2.GetRasterBand(x)
            array = band.ReadAsArray().astype(np.float)

        dem=dem2.GetRasterBand(dem2.RasterCount).ReadAsArray().astype(np.float)

        dem[dem<0]=0 #eliminate non values
        maxDem=np.max(dem)
        gray = cv2.GaussianBlur(dem,(5,5),0)

        treeTops = cv2.imread(argv[3],0)
        if treeTops is None:raise Exception("demUtils no treeTops at "+str(argv[3]))

        grayForPercentile=gray.copy()
        grayForPercentile[grayForPercentile==0]=np.nan

        thresholdList=[]
        iList=[]
        for i in range(10,100,10):
            iList.append(i)
            thresholdList.append(np.nanpercentile(grayForPercentile, i))

        i=0
        while i<len(thresholdList):
            thisPercDict=averageTopRegionSizePercentage(dem,treeTops,thresholdList[i])
            #print("tree tops remaining at threshold "+str(th)+" "+"{:.2f}".format(erasedTops)+" with remaining pixel perc "+"{:.2f}".format(erasedPixPerc),flush=True)
            #print("Percentile: "+str(10*(i+1))+" : "+str(thisPercDict),flush=True)
            avPoints=0
            avArea=0
            for k,v in thisPercDict.items():
                if k!=0:
                    avPoints+=v[0]
                    avArea+=v[1]
            numRegions=len(thisPercDict)-1
            if 0 in thisPercDict:totalPoints=avPoints+thisPercDict[0][0]
            else:totalPoints=avPoints
            print(" Perc: "+str(iList[i])+" height "+"{:.2f}".format(thresholdList[i])+" total points "+str(totalPoints)+" nonbck points "+str(avPoints/totalPoints)+" av points "+"{:.2f}".format(avPoints/numRegions)+" av area "+"{:.2f}".format(avArea/numRegions)+" numRegions "+str(numRegions))
            i+=1
    elif (int(argv[1]))==7: # top by top, paint a big circle, find the tallest point in it and compare to the annotated point (altitude and distance).
        dem = cv2.imread(argv[2],cv2.IMREAD_UNCHANGED)
        if dem is None:
            raise Exception(str(argv[2])+ "not found ")

        realTops = cv2.imread(argv[3],cv2.IMREAD_GRAYSCALE)
        if realTops is None:
            raise Exception(str(argv[3])+ "not found ")

        # compute conected components in tops image
        maskImage=realTops.copy()
        mask = cv2.threshold(255-realTops, 10, 255, cv2.THRESH_BINARY)[1]

        #compute connected components
        numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

        avDif = 0
        avDist = 0
        newTops = []
        # for every top, find maximum and distance to it
        for i in range(1,numLabels):
            left = int(stats[i,cv2.CC_STAT_LEFT])
            top = int(stats[i,cv2.CC_STAT_TOP])
            width = int(stats[i,cv2.CC_STAT_WIDTH])
            area = int(stats[i,cv2.CC_STAT_AREA])
            height = int(stats[i,cv2.CC_STAT_HEIGHT])

            thisTopHeight = dem[int(centroids[i][1]),int(centroids[i][0])]
            # IS this really the right window? (looks like it)
            auxDem = dem[top:top+height,left:left+width].copy()
            auxTops = mask[top:top+height,left:left+width].copy()

            auxauxDem = auxDem.copy()

            #cv2.imwrite(str(i)+"tops.jpg",auxTops)

            #auxDem[auxTops==0] = 0

            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(auxDem)

            center = (centroids[i][1]-top,centroids[i][0]-left)
            centerHeight = auxDem[int(center[0]),int(center[1])] # we already swaped dimensions in the line above

            print("top is at "+str(thisTopHeight)+" and max is at "+str(maxVal)+" and center is at "+str(centerHeight))
            thisDif = maxVal - thisTopHeight

            if thisDif <0 or centerHeight != thisTopHeight:raise Exception("something went wrong "+str(thisDif))
            else: avDif += thisDif

            #check maxloc correspondence
            maxPos = (int(maxLoc[0]),int(maxLoc[1]))
            maxPosAbsolute = (left + maxPos[0],top + maxPos[1])

            print("max val "+str(maxVal)+" shoudl be the same as "+str(auxDem[maxPos[1],maxPos[0]])+" and "+str(dem[maxPosAbsolute[1],maxPosAbsolute[0]]))

            #print(str(maxLoc)+" and the centroid "+str(center))
            #print(distPoints(maxLoc,center))
            avDist += distPoints(maxLoc,center)

            #print(str(thisTopHeight)+" and "+str(auxDem[int(center[0]),int(center[1])]))
            newTops.append(((maxPosAbsolute[0],maxPosAbsolute[1])))
            #podria ser tb amb el top i el left girats
            #print(str(centroids[i])+" and "+str(newTops[-1]) )


            # show things, circles on top and max
            """
            auxauxDem = 255*(auxauxDem - minVal)/(maxVal-minVal)
            auxDem3 = auxauxDem.copy()
            cv2.circle(auxauxDem,(int(center[0]),int(center[1])),2,0,-1)
            cv2.circle(auxauxDem,(int(maxLoc[1]),int(maxLoc[0])),1,0,-1)

            newSize = 5
            if auxauxDem.shape[0]<newSize:
                cv2.imwrite(str(i)+"demPoints.jpg",cv2.resize(auxauxDem,(500,500)))
                cv2.imwrite(str(i)+"dem.jpg",cv2.resize(auxDem3,(500,500)))
            else:
                cv2.imwrite(str(i)+"demPoints.jpg",auxauxDem)
                cv2.imwrite(str(i)+"dem.jpg",auxDem3)
            """

        print("average altitude distance "+str(avDif/(numLabels-1)))
        print("average distance "+str(avDist/(numLabels-1)))

        #now paint corrected tops
        circleSize=5
        maskImage=realTops.copy()
        maskImage[maskImage<255]=255

        for seed in newTops:
            cv2.circle(maskImage, (int(seed[0]),int(seed[1])), circleSize, 20, -1)

        cv2.imwrite(argv[4],maskImage)

    elif (int(argv[1]))==8: # divide in windows, at each window threshold by percentile and erode significantly. Then globally count lost tops


        dem = cv2.imread(argv[2],cv2.IMREAD_UNCHANGED)
        if dem is None:raise Exception("demUtils no DEM at "+str(argv[2]))


        treeTops = cv2.imread(argv[3],0)
        if treeTops is None:raise Exception("demUtils no treeTops at "+str(argv[3]))

        dem[dem<0]=0 #eliminate non values

        #storePrettyDEM(dem,"origDEM.jpg")

        size = int(argv[4])
        perc = int(argv[5])
        erokerns = int(argv[6])
        eroIts = int(argv[7])

        minNumPointsTree = 50

        for (x, y, window) in sliding_window(dem, stepSize=size, windowSize=(size, size),allImage=False):

            if np.sum(window>0) > minNumPointsTree:
                dem[y:y + size, x:x + size] = binarizeWindowReturnDEM(window,perc,erokerns,eroIts)

        dem = storePrettyDEM(dem,"cleanDEM.jpg")

        dem[dem>100] = 255

        print("tree tops lost in dem "+str(countLostTopsRegionSize(dem.astype("uint8"),treeTops)))


    else:
        raise Exception("demUtils, wrong code")

if __name__ == '__main__':
    #labelIMResample(sys.argv[1])
    main(sys.argv)
