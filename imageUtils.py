import numpy as np
import cv2
import sys


from math import sqrt


def sliding_window(image, stepSize, windowSize, allImage=False):
    if allImage:
        yield(0,0,image[:,:])
    else:
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


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


stupidCount = 0

# Given a window, eliminate possible outliers and get only the top pixels
def binarizeWindowReturnDEM(win, lowerPerc = 20, eroKernS = 2, eroIt = 3):

    #global stupidCount
    #stupidCount+=1

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

    #cv2.imwrite("./shit/"+str(stupidCount)+"eroded.jpg",erosion)

    return erosion


# Given a window, eliminate possible outliers and get only the top pixels
def binarizeWindow(win, lowerPerc = 40, eroKernS = 3, eroIt = 10):

    #global stupidCount
    #stupidCount+=1

    lowerPerc = 40
    higherPerc = 99
    fromRatio = 0.02

    winRet = win.copy()
    winPerc = win.copy()
    winPerc[winPerc==0]=np.nan
    minWin = np.nanpercentile(winPerc,lowerPerc)
    maxWin = np.nanpercentile(winPerc,higherPerc)
    #print("slidingWindow, binarizeWindow, window height "+str(minWin)+" "+str(minWin+ (maxWin-minWin)*fromRatio)+" "+str(maxWin))

    winRet[win>maxWin] = maxWin
    winRet[win<minWin + (maxWin-minWin)*fromRatio] = 0

    #cv2.imwrite(str(stupidCount)+"noteroded.jpg",winRet)

    winRet[winRet>0] = 255

    erosionKernel=np.ones((eroKernS,eroKernS),np.uint8)
    erosion=cv2.erode(winRet,erosionKernel,iterations = eroIt)

    winRet[winRet>0] = 255

    #cv2.imwrite(str(stupidCount)+"eroded.jpg",erosion)

    return winRet


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
    main(sys.argv)
