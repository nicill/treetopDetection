import numpy as np
import cv2
import sys
import crownSegmenterEvaluator as CSE
from osgeo import gdal

from scipy.spatial.distance import directed_hausdorff
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
def countLostTopsRegionSize(dem, treeTops,minSize=10350):

    mask = cv2.threshold(255-treeTops, 80, 255, cv2.THRESH_BINARY)[1]
    #compute connected components
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

    dem[dem>0]=255
    numLabels2, labelImage2,stats2, centroids2 = cv2.connectedComponentsWithStats(dem)

    count=0
    for i in range(1,len(centroids)):
        x=centroids[i][1]
        y=centroids[i][0]

        #print(x)

        currentLabel=labelImage2[int(x)][int(y)]
        if currentLabel==0:count+=1
        else:
            pointsThisLabel=stats2[currentLabel,cv2.CC_STAT_AREA]
            if pointsThisLabel<minSize:
                #print("had "+str(pointsThisLabel))
                count+=1

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

def thresholdDEMBinarised(dem,th1,th2):
    #print("thresholding at "+str(th1)+" "+str(th2))
    #threshold DEM
    demAux=dem.copy()

    demAux[dem<th1]=0
    demAux[dem>th2]=0
    demAux[demAux!=0]=255
    #print("                bigger than zero "+str(np.sum(demAux>0)))

    return demAux.astype("uint8")

def isInLayer(center,layer):
    return layer[int(center[0]),int(center[1])]==255


def main(argv):

    #argv[1] contains mode    cv2.imwrite("hkl.jpg",demAux)

    if (int(argv[1]))==0: # count how many tree tops are in the floor, argv[1] contains the DEM and argv[2] the treetops

        dem = cv2.imread(argv[2],0)
        if dem is None:raise Exception("demUtils no DEM at "+str(argv[2]))

        treeTops = cv2.imread(argv[3],0)
        if treeTops is None:raise Exception("demUtils no treeTops at "+str(argv[3]))

        print("tree tops lost in dem "+str(countLostTops(dem,treeTops)))
    elif (int(argv[1]))==1: # paint the treeTops bigger

        dem = cv2.imread(argv[2],0)
        if dem is None:raise Exception("demUtils no DEM at "+str(argv[2]))

        treeTops = cv2.imread(argv[3],0)
        if treeTops is None:raise Exception("demUtils no treeTops at "+str(argv[3]))

        maskImage=treeTops.copy()
        mask = cv2.threshold(255-treeTops, 80, 255, cv2.THRESH_BINARY)[1]
        #compute connected components
        numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

        circleSize=20
        for seed in centroids[1:]:
            cv2.circle(maskImage, (int(seed[0]),int(seed[1])), circleSize, 20, -1)

        cv2.imwrite(argv[4],maskImage)

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


    else:
        raise Exception("demUtils, wrong code")

if __name__ == '__main__':
    main(sys.argv)
