# This method takes the point coordinates of each tree top given by crownSegementerEvaluator and extracts
# a squared patch arround each one. Then, uses the classified masks of the mosaics to know in which species belongs.
# Then stores each small labeled patch in a folder.

import sys
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple
import multiLabelDataAugmenter as aug

mosaicInfo = namedtuple("mosaicInfo","path mosaicFile mosaicTopsFile numClasses layerNameList layerFileList outputFolder " )

def interpretParameters(paramFile,verbose=False):
	# read the parameter file line by line
	f = open(paramFile, "r")
	patchSize=-1
	mosaicDict={}
	important=[]
	unImportant=[]

	for x in f:
		lineList=x.split(" ")
		# read every line
		first=lineList[0]

		if first[0]=="#": #if the first character is # treat as a comment
			if verbose:print("COMMENT: "+str(lineList))
		elif first=="\n":# account for blank lines, do nothing
			pass
		elif first=="patchSize":
			patchSize=int(lineList[1].strip())
			if verbose:print("Read Patch Size : "+str(patchSize))
		elif first=="mosaic":
			layerNameList=[]
			layerFileList=[]

			# read the number of layers and set up reading loop
			filePath=lineList[1]
			mosaic=lineList[2]
			mosaicTops = lineList[3]
			numClasses=int(lineList[4])
			outputFolder=lineList[5+numClasses*2].strip()
			for i in range(5,numClasses*2+4,2):
				layerNameList.append(lineList[i])
				layerFileList.append(filePath+lineList[i+1])

			#make dictionary entry for this mosaic
			mosaicDict[mosaic]=mosaicInfo(filePath,mosaic,mosaicTops,numClasses,layerNameList,layerFileList,outputFolder)
			if verbose:
				print("\n\n\n")
				print(mosaicDict[mosaic])
				print("\n\n\n")
				#print("Read layers and file : ")
				#print("filePath "+filePath)
				#print("mosaic "+mosaic)
				#print("num Classes "+str(numClasses))
				#print("layerName List "+str(layerNameList))
				#print("layer List "+str(layerList))
				#print("outputFolder "+outputFolder)
		elif first=="important":
			for x in lineList[1:]:
				if x.strip() not in important:important.append(x.strip())
		elif first=="unimportant":
			for x in lineList[1:]:
				if x.strip() not in unImportant:unImportant.append(x.strip())
		else:
			raise Exception("ImagePatchAnnotator:interpretParameters, reading parameters, received wrong parameter "+str(lineList))

		if verbose:(print(mosaicDict))

	return patchSize,mosaicDict,important,unImportant


def borderPoint(image,point):
	margin=100
	top1=image.shape[0]
	top2=image.shape[1]

	return point[0]<margin or (top1-point[0])<margin or point[1]<margin or (top2-point[1])<margin

# Function to take a binary image and output the center of masses of its connected regions
# THIS METHOD IS A COPY OF crownSectmenterEvaluator method! must be deleted!!!
# Function to take a binary image and output the center of masses of its connected regions
def listFromBinary(fileName,ROIFILE=None):

    #open filename
    im=cv2.imread(fileName,cv2.IMREAD_GRAYSCALE)
    #print(im.shape)
    if im is None: return []
    else:
        mask = cv2.threshold(255-im, 40, 255, cv2.THRESH_BINARY)[1]

        #compute connected components
        numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)
        #print("crownSegmenterEvaluator, found "+str(numLabels)+" "+str(len(centroids))+" points for file "+fileName)

        #im2 = 255 * np. ones(shape=[im.shape[0], im.shape[1], 1], dtype=np. uint8)

        #print(" listFromBinary, found  "+str(len(centroids)))
        #print(centroids)

        newCentroids=[]
        ROI=cv2.imread(ROIFILE,cv2.IMREAD_GRAYSCALE)
        if ROI is None: newCentroids=centroids
        else:
            ROI[ROI<10]=0
            for c in centroids:
                if pointInROI(ROI,c):newCentroids.append(c)

        #print(" listFromBinary, refined  "+str(len(newCentroids)))
        #print(newCentroids)

        return newCentroids[1:]

def pointInROI(ROI,p):
    return ROI[int(p[1])][int(p[0])]==0


def getSquareList(w_size, p, img,numSquares):
	#the first one is always centered
	returnList=[getSquare(w_size,p,img)]
	for i in range(1,numSquares):
		# generate new center
		halfSize=w_size//2
		xOffset=random.randint(-halfSize,halfSize)
		yOffset=random.randint(-halfSize,halfSize)

		#append to the list
		returnList.append(getSquare(w_size,(p[0]+xOffset,p[1]+yOffset),img))

	return returnList

def getSquare(w_size, p, img):


	height, width, channels = img.shape

	#isInside = (int(p[0])-w_size//2) >= 0 and (int(p[0])+w_size//2) < width and (int(p[1])-w_size//2) >= 0 and (int(p[1])+w_size//2) < height
	isInside = (int(p[1])-w_size//2) >= 0 and (int(p[1])+w_size//2) < width and (int(p[0])-w_size//2) >= 0 and (int(p[0])+w_size//2) < height

	assert isInside, "The required window is out of bounds of the input image "+str(p)

	return img[int(p[0])-w_size//2:int(p[0])+w_size//2, int(p[1])-w_size//2:int(p[1])+w_size//2]

def isInLayer(center,layer):
	return layer[int(center[0]),int(center[1])]==255


def main(argv):

	#define whether or not we will use uncentered tops
	uncentered=True
	if uncentered: uncenteredDict={"deciduous":3,"sickfir":5,"healthyfir":3}
	else:uncenteredDict={"deciduous":1,"sickfir":1,"healthyfir":1}
	print(uncenteredDict)

	augment=int(argv[2])>0
	print("augment "+str(augment)+" "+str(argv[2]))
	augmentFactor=0
	if augment: augmentFactor=int(argv[2])
	numberOfAugmentations=6

	decreaseFactor=25 #over 100, a random draw will be made between 1 and 100 and if the result is smaller than this factor the image will not be saved

	# Add new mode "ballanced", receive as input (or hardcode) a weight vector (adding to 1)
	# then count number of images per class. let n be the number of images in the bigger class, then numClasses*n equals 100 and the classes that need to be augmented are augmented.

	try:
		# verbose = False
		patchSize, mosaicDict, important, unImportant = interpretParameters(argv[1])

		if augment:
			print("Important classes to be augmented "+str(important)+" with factor "+str(augmentFactor))
			print("Unimportant classes to be decreased "+str(unImportant)+" with factor "+str(decreaseFactor))

		#if verbose: print(mosaicDict)
		counter = 0

		for mosaicName, mosaicInfo in mosaicDict.items():
			noClassCounter=0
			roiFileName=mosaicInfo.path +mosaicName[:-4]+"ROI.jpg"
			#print(roiFileName)

			print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ "+str(mosaicName))

			mosaicTopsFile = mosaicInfo.path + mosaicInfo.mosaicTopsFile
			mosaicFile = mosaicInfo.path + mosaicInfo.mosaicFile
			outputFolder = mosaicInfo.path + mosaicInfo.outputFolder + "/"

			# if verbose: print("\n\nstarting processing of first mosaic and layers "+str(v)+"\n\n")
			treetops_mask = cv2.imread(mosaicTopsFile, cv2.IMREAD_GRAYSCALE)
			if treetops_mask is None: raise Exception("treetop_mask not read "+mosaicTopsFile)

			mosaic = cv2.imread(mosaicFile, cv2.IMREAD_COLOR)
			if mosaic is None: raise Exception("mosaic not read "+mosaicFile)

			centroids = listFromBinary(mosaicTopsFile,roiFileName)

			layers=[]
			for layerFileName in mosaicInfo.layerFileList:
				aux=cv2.imread(layerFileName, cv2.IMREAD_GRAYSCALE)
				if aux is None: raise Exception(" Missing some layer "+str(layerFileName))

				layers.append(255-aux)

			for cent in centroids:

				try:
					className="EMPTYCLASS"
					for i in range(mosaicInfo.numClasses):
						#print(str((cent[1],cent[0]))+" TO BE CHECKED FOR CLASS "+mosaicInfo.layerNameList[i])

						if isInLayer((cent[1],cent[0]),layers[i]):
							if className!="EMPTYCLASS":
								raise Exception(str((cent[1],cent[0]))+"center belongs to two classes,  "+className+" and "+mosaicInfo.layerNameList[i])
							#print("found that "+str((cent[1],cent[0]))+" belongs to "+mosaicInfo.layerNameList[i])
							className=mosaicInfo.layerNameList[i]

					if className=="EMPTYCLASS":
						#raise Exception(str((cent[1],cent[0]))+"center belongs to no class")
						print(str((int(cent[0]),int(cent[1])))+"center belongs to no class "+str(noClassCounter))
						noClassCounter+=1
					else:
						# found a center in a class, now, first make as many images as indicated in
						# opencv works with inverted coords, so we have to invert ours.
						squareList = getSquareList(patchSize, (cent[1],cent[0]), mosaic,uncenteredDict[className])

						bool=className in important
						#print("bool "+str(bool)+" augment "+str(augment)+" importnat "+str(important)+" class name "+className )
						if augment and className in important:
							#print("augmenting!!!!!!")
							# First, store the original
							for square in squareList:
								cv2.imwrite(outputFolder+"SP"+className+"PATCH"+str(counter)+".jpg", square)
								counter+=1

							#augmentation loop
							for x in range(augmentFactor):
								square=random.choice(squareList)
								aug.augment(square,random.randint(0,numberOfAugmentations-1),outputFolder+"SP"+className+"PATCHAUGMENTED"+str(counter)+".jpg",False)
								counter+=1
						elif augment and className in unImportant:
							#print("unimportant !!!!!!")
							if random.randint(1,100)>decreaseFactor:
								for square in squareList:
									cv2.imwrite(outputFolder+"SP"+className+"PATCH"+str(counter)+".jpg", square)
									counter+=1
						else:
							for square in squareList:
								cv2.imwrite(outputFolder+"SP"+className+"PATCH"+str(counter)+".jpg", square)
								#print("Writing "+outputFolder+"SP"+className+"PATCH"+str(counter)+".jpg")

								counter+=1

				except AssertionError as error:
					print(error)


	except AssertionError as error:

		print(error)


# Exectuion example -> python treeTopPatcher.py <path_to_params_file>

if __name__ == "__main__":
	main(sys.argv)
