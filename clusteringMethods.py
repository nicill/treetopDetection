import numpy as np
import cv2
# Clustering imports
from sklearn.cluster import KMeans, SpectralClustering
import skfuzzy as skf
from sklearn.mixture import GaussianMixture
from skimage.transform import pyramid_expand, pyramid_reduce, resize

from sklearn.cluster import DBSCAN

from sklearn.cluster import MeanShift, estimate_bandwidth


def dbscan(im,eps,minS=10):
    labimg = cv2.medianBlur(im,5)
    n = 0
    while(n<4):
        labimg = cv2.pyrDown(labimg)
        n = n+1

    #feature_image=np.reshape(labimg, [-1, 3])
    rows, cols = labimg.shape

    db = DBSCAN(eps=eps, min_samples=minS, metric = 'euclidean',algorithm ='auto')
    db.fit(labimg.flatten().reshape(rows*cols, 1))
    labels = db.labels_
    nClusters=len(set(labels)) - (1 if -1 in labels else 0)
    return computeClusterMaxoids(im,np.reshape(labels, [rows, cols]),nClusters)


def k_means_clustering(inp_image, n_clusters=2):
    if inp_image is None:
        print("Empty Input. Exiting")
        return None
    # Create K Means Model
    k_means = KMeans(n_clusters=n_clusters)
    shape = inp_image.shape
    # Fit on Input Image
    k_means.fit(inp_image.flatten().reshape(shape[0]*shape[1], 1))
    # Get Cluster Labels
    #print(str(k_means.cluster_centers_))
    #return k_means.cluster_centers_.reshape((-1, 2))
    clust = k_means.labels_
    #print(str(clust))
    #return computeClusterMedoids(clust.reshape(shape[0], shape[1]),n_clusters)
    return computeClusterMaxoids(inp_image,clust.reshape(shape[0], shape[1]),n_clusters)

def fuzzy_c_means(inp_image, n_clusters=2):
    if inp_image is None:
        print("Empty Input. Exiting")
        return

    shape = inp_image.shape
    # Create and Train on FCM Model
    centers, u, u0, d, jm, n_iters, fpc = skf.cluster.cmeans(
        inp_image.flatten().reshape(shape[0]*shape[1], 1).T,
        c=n_clusters,
        m=float(3),
        error=float(0.05),
        maxiter=int(100),
        init=None,
        seed=int(21)
    )
    # Get Cluster Labels with Max Probability
    clust = np.argmax(u, axis=0)

    return computeClusterMaxoids(inp_image,clust.reshape(shape[0], shape[1]),n_clusters)

def gaussian_mixture_model(inp_image, n_clusters=2):
    shape = inp_image.shape
    inp_image = inp_image.flatten().reshape(shape[0]*shape[1], 1)
    MAX_ITER=25
    RANDOM_STATE=21
    COVARIANCE_TYPE="full"
    #WINDOW_SIZE=3
    # Create Gaussian Mixture Model with Config Parameters
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type=COVARIANCE_TYPE,
        max_iter=MAX_ITER, random_state=RANDOM_STATE)
    # Fit on Input Image
    gmm.fit(X=inp_image)
    # Get Cluster Labels
    clust = gmm.predict(X=inp_image)

    return computeClusterMaxoids(inp_image.reshape(shape[0], shape[1]),clust.reshape(shape[0], shape[1]),n_clusters)

# Seems to run,but veeeery slow
def meanShift(originImg):
    #source https://stackoverflow.com/questions/46392904/scikit-mean-shift-algorithm-returns-black-picture
    # Shape of original image
    originShape = originImg.shape

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities
    flatImg=np.reshape(originImg, [-1, 1])

    # Estimate bandwidth for meanshift algorithm
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True,n_jobs=5,min_bin_freq=5)

    # Performing meanshift on flatImg
    ms.fit(flatImg)

    # (r,g,b) vectors corresponding to the different clusters after meanshift
    labels=ms.labels_

    # Remaining colors after meanshift
    cluster_centers = ms.cluster_centers_

    # Finding and diplaying the number of clusters
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters)

    # Displaying segmented image
    segmentedImg = np.reshape(labels, originShape[:2])

    return computeClusterMaxoids(originImg,segmentedImg,n_clusters)

def computeClusterMaxoids(inpImage,clust,nClusters):#clust is shaped as the image with labels
    # This function can be modified pretty simply so it also
    # receives a minimum label where to start the label numeration for the current window
    # Reenumerates the labels so they start at this minimum
    # Is there a "background" label?
    # This would be done to take advantage of the segmentations produced by clustering methods

    coordList=[]
    maxoids=[]
    for i in range(nClusters): maxoids.append([0,0,0])
    for x in range(clust.shape[0]):
        for y in range(clust.shape[1]):
            #print(str(x)+" "+str(y)+" "+str(clust[x][y]))
            if (clust[x][y]>0): # some algorithms output -1 values that are not really regions
                #print(inpImage[x,y])
                maxX=maxoids[clust[x][y]][0]
                maxY=maxoids[clust[x][y]][1]
                #print(inpImage[maxX,maxY])
                if inpImage[x,y]> inpImage[maxX,maxY]:
                    maxoids[clust[x][y]][0]=x
                    maxoids[clust[x][y]][1]=y
                maxoids[clust[x][y]][2]+=1
            #print("                               "+str(medoids))

    for c in range(nClusters):
        if maxoids[c][2]!=0 :
            coordList.append((maxoids[c][0],maxoids[c][1]))
            #print("IN here! "+str(coordList[-1]))

    return coordList
