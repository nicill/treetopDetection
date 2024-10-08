# File to evaluate the result of crown segmentation methods.
# First, a function that receives two binary masks with the treetops as small circles
# Then transforms them into a list of points and then computes hausdorf distance between the point segmentations

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,50))
import cv2

import sys
from skimage import measure
import matplotlib.pyplot as plt

from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import KDTree
import numpy as np
from collections import defaultdict

def borderPoint(image,point):
    margin=500
    top1=image.shape[0]
    top2=image.shape[1]

    return point[0]<margin or (top1-point[0])<margin or point[1]<margin or (top2-point[1])<margin

def pointInROI(ROI,p):
    return ROI[int(p[1])][int(p[0])]==0
    #return True

# Function to take a binary image and output the center of masses of its connected regions
def listFromBinary(fileName,ROIFILE=None,deleteSmall=None):

    #open filename
    im=cv2.imread(fileName,cv2.IMREAD_GRAYSCALE)
    #print(im.shape)
    if im is None:
        #print("filename, "+str(fileName))
        return []
    else:
        mask = cv2.threshold(255-im, 40, 255, cv2.THRESH_BINARY)[1]

        #first, eliminate all treetops outside of the ROI
        """
        if ROIFILE is not None:
            ROI=cv2.imread(ROIFILE,cv2.IMREAD_GRAYSCALE)
            if ROI is not None:
                ROI[ROI<50]=0
                ROI[ROI>50]=255
                mask[ROI!=0]=0
                #cv2.imwrite("papap.jpg",mask)
                #print("FILE "+fileName)
                #sys.exit()
        """

        #clean up small regions
        if ROIFILE is not None and deleteSmall is not None:

            ROI=cv2.imread(ROIFILE,cv2.IMREAD_GRAYSCALE)
            if ROI is not None:
                ROI[ROI<50]=0
                ROI[ROI>50]=255
                mask[ROI!=0]=0

            #compute connected components
            numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)
            #print("crownSegmenterEvaluator, found "+str(numLabels)+" "+str(len(centroids))+" points for file "+fileName)


            minTopSize=deleteSmall
            #print("doing con comp of "+str(fileName))
            newCentroids=[]
            index=1
            for l in range(1,numLabels):
                currentCount=np.count_nonzero(labelImage==l)
                if currentCount>minTopSize:newCentroids.append(centroids[index])
                index+=1
            #print("output "+str(len(newCentroids)))
        else:

            #compute connected components
            numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)
            #print("crownSegmenterEvaluator, found "+str(numLabels)+" "+str(len(centroids))+" points for file "+fileName)


            newCentroids=[]
            if ROIFILE is not None:
                ROI=cv2.imread(ROIFILE,cv2.IMREAD_GRAYSCALE)
                if ROI is None:
                #    print("roi in "+str(ROIFILE)+" was none")
                    newCentroids=centroids
                else:
                    ROI[ROI<50]=0
                    for c in centroids[1:]:
                        if pointInROI(ROI,c):newCentroids.append(c)
            else:
                newCentroids=centroids

        #print("list form binary outputting "+str(len(newCentroids)))
        return newCentroids

def labelImEval(list1, labelIm):
    """
    Receives a label image with the cannopies separated by codes
    and a list of tops
    and returns the percentage of canopies detected, the percentage
    of predicted points that repeat a canopy
    the percentafe of canopies missed and
    the percentage of predicted points out of a canopy (over total canopies)
    """
    def procTop(c):
        x,y = c
        detCpy[labelIm[int(y),int(x)]]+=1
    detCpy = defaultdict(lambda: 0)
    list(map(procTop,list1))
    allLabels = np.unique(labelIm)
    totalLabels = len(allLabels)-1
    labelsFound = len(detCpy.keys()) - 1 if 0 in detCpy else len(detCpy.keys())
    labelsMissed = totalLabels - labelsFound - 1
    pointsInBackground = detCpy[0]
    repeatedPoints = len(list1) - labelsFound - pointsInBackground
    print("total trees "+str(totalLabels))
    print("found "+str(labelsFound))
    print("missed "+str(labelsMissed))
    print("repeated "+str(repeatedPoints))
    print("bck "+str(pointsInBackground))
    return 100*(labelsFound/totalLabels),100*(repeatedPoints/totalLabels),100*(labelsMissed/totalLabels),100*(pointsInBackground/totalLabels)

def maskFromPointList(file, mask,dest):
    """
    create a binary point image from the list of the bounding boxes
    """
    outIM = mask.copy()
    outIM[mask>0] = 0 # black image with the right dimensions
    with open(file) as f:
        for line in f:
            contents = line.strip().split(" ")
            #print(contents)
            c0 = float(contents[0]) + (float(contents[2])-float(contents[0]))/2
            c1 = float(contents[1]) + (float(contents[3])-float(contents[1]))/2
            cv2.circle(outIM, (int(c0),int(c1)), 5, 255, -1)
            #print("circle "+str(c0)+" "+str(c1) )
    cv2.imwrite(dest,255-outIM)


def hausdorfDistance(u,v): # computes Hausdorf distance between two lists of point
    if len(u)==0 or len(v)==0: return -1
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

def matchedPercentage(list1,list2,epsilon,mode=0):
    if len(list1)==0 or len(list2)==0: return -1

    newList1=np.asarray([[x,y] for x,y in list1 ])
    newList2=np.asarray([[x,y] for x,y in list2 ])

    #print("matched perc, first list "+str(len(newList1)))
    #print("matched perc, second list "+str(len(newList2)))

    kdt = KDTree(newList2, leaf_size=30, metric='euclidean')
    dist,ind=kdt.query(newList1, k=1)
    #print(dist)
    count=0
    for d in dist:
        if d<epsilon:count+=1

    if mode==0:return 100*(count/len(list1))
    elif mode ==1: return count
    elif mode==2: return (count,100*(count/len(list1)))
    else: raise Exception(" crownSegmenterEvaluator matching percentage, wrong mode!!!")

def avEucDistance(list1,list2):
    if len(list1)==0 or len(list2)==0: return -1

    newList1=np.asarray([[x,y] for x,y in list1 ])
    newList2=np.asarray([[x,y] for x,y in list2 ])

    kdt = KDTree(newList2, leaf_size=30, metric='euclidean')
    dist,ind=kdt.query(newList1, k=1)
    sum=0
    for d in dist:
        if len(d)>0:sum+=d[0]

    return sum/len(list1)

def avEucDistanceMatched(list1,list2,epsilon):
    #print("lists")
    #print(len(list1))
    #print(len(list2))

    if len(list1)==0 or len(list2)==0: return -1

    newList1=np.asarray([[x,y] for x,y in list1 ])
    newList2=np.asarray([[x,y] for x,y in list2 ])

    kdt = KDTree(newList2, leaf_size=30, metric='euclidean')
    dist,ind=kdt.query(newList1, k=1)
    sum=0
    totalPoints=0
    for d in dist:
        if len(d)>0 and d[0]<epsilon:
            sum+=d[0]
            totalPoints+=1

    return sum/totalPoints

def countRepeatedMatched(list1,list2,epsilon):
    #count how many points in list1 are matched to repeated points in list2
    if len(list1)==0 or len(list2)==0: return -1

    newList1=np.asarray([[x,y] for x,y in list1 ])
    newList2=np.asarray([[x,y] for x,y in list2 ])

    kdt = KDTree(newList2, leaf_size=30, metric='euclidean')
    dist,ind=kdt.query(newList1, k=1)

    indexDict={}
    repeatedPoints=0
    totalPoints=0
    for i in range(len(dist)):
        if dist[i]<epsilon:#matched point
            totalPoints+=1
            if ind[i][0] in indexDict:repeatedPoints+=1
            else:indexDict[ind[i][0]]=True

#    print("total matched "+str(totalPoints))
#    print("total repeated "+str(repeatedPoints))

    return 100*repeatedPoints/totalPoints

def main(argv):
    # argv[1] contains the distance method (0 hausdorff, 1, matched point percentage)
    # argv[2],argv[3] contains the names of the files with the first and second mask
    # Further parameters may contain specific information for some methods

    option=int(argv[1])
    file1=argv[2]
    file2=argv[3]
    #print("ARGV")
    #print(argv)
    if(len(argv)>4):ROIFile=argv[4]
    else:
    #    print("from the start, roi file is NONE")
        ROIFile=None

    #first, turn the binary masks of files 1 and 2 into lists of points
    #print("List1 ")
    list1=listFromBinary(file1,ROIFile)
    list2=listFromBinary(file2,ROIFile)

    # Now, compute the distance between sets indicated by the option
    if option == 0: # compute hausdorff distance between two masks
        # Second, compute hausdorff distance
        print(format(hausdorfDistance(list1,list2),'.2f'),end=" ")
    elif option==1: #number of matched points, in this case we need one extra parameter epsilon
        epsilon=float(argv[5])
        # The first file must be the ground truth
        print(format(matchedPercentage(list1,list2,epsilon),'.2f'),end=" ")
    elif option==2: # point difference
    # the ground truth file should be the first
        realPointsNumber=len(list1)
        predictedPointsNumber=len(listFromBinary(file2,ROIFile,12))
        #print("real "+str(realPointsNumber)+" predicted "+str(predictedPointsNumber))
        print(format(100*(realPointsNumber-predictedPointsNumber)/realPointsNumber,'.2f'),end=" ")
    elif option==3:
        #simply count point
        print(str(len(list1))+" "+str(len(list2)))
    elif option==4:
        #return average euclidean distance
        print(format(avEucDistance(list1,list2),'.2f'),end=" ")
    elif option==5:
        #return average euclidean distance between matched points
        epsilon=float(argv[5])
        print(format(avEucDistanceMatched(list1,list2,epsilon),'.2f'),end=" ")
    elif option==6:
        #return average euclidean distance between matched points
        epsilon=float(argv[5])
        print(format(countRepeatedMatched(list1,list2,epsilon),'.2f'),end=" ")

    else: raise ("crownSegmenterEvaluator, Wrong option")


if __name__ == "__main__":
    #maskFromPointList(sys.argv[1],cv2.imread(sys.argv[2],0),sys.argv[3])
    print(labelImEval(listFromBinary(sys.argv[1]),cv2.imread(sys.argv[2],cv2.IMREAD_UNCHANGED)))
    #main(sys.argv)
