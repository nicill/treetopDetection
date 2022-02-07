import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils import contours
#import pickle
#from keras.models import load_model
#from sklearn.preprocessing import OneHotEncoder
import os
import sys
import sliding_window as sw

def prep(thresh,args):
	#minNumPointsTree=10350

	minNumPointsTree=int(args["minPointsTree"])

	#https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	try:cnts = contours.sort_contours(cnts)[0]
	except: cnts=[]

	#not paying any atention to what safonova did
	tops=[]
	for c in cnts:
		#compute area of the contour
		area = cv2.contourArea(c)

		if (area>minNumPointsTree):

			# compute the center of the contour
			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])

			tops.append((int(cY),int(cX)))
			#tops.append((cY,cX))

		#else:
		#	print("small contour "+str(area))
	#print(tops)
	return tops

def refineWithDEM(dem,tops,minSize=2500):
	dem[dem>0]=255
	numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(dem)
	retTops=[]
	for top in tops:
		x=top[1]
		y=top[0]

		currentLabel=labelImage[int(x)][int(y)]
		pointsThisLabel=stats[currentLabel,cv2.CC_STAT_AREA]
		if currentLabel!=0 and pointsThisLabel>minSize: retTops.append(top)

	return retTops

def main(argv):

	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="Path to image")
	ap.add_argument("-d", "--dem", required=False, help="Path to Digital Elevation Map (DEM)")
	ap.add_argument("-s", "--size", required=True, help="Window size (squared)")
	ap.add_argument("-th", "--threshold", required=True, help="Image thresholding value")
	ap.add_argument("-mp", "--minPointsTree", required=True, help="Image thresholding value")
	ap.add_argument("-o", "--binOut", required=False, help="Path of the resulting binary image")
	args = vars(ap.parse_args())

	print(args["image"])

	frame = cv2.imread(args["image"])
	if frame is None: raise Exception("frame not read")

	if args["size"] is None: raise Exception("no window size providee\d")
	size=int(args["size"])

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)
	thresh = cv2.threshold(blurred, int(args["threshold"]), 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations = 16)
	thresh = cv2.equalizeHist(thresh)
	thresh = cv2.dilate(thresh, None, iterations = 12)


	seeds=[]
	for (x, y, window) in sw.sliding_window(thresh, stepSize=size, windowSize=(size, size),allImage=False):
		thisWindowSeeds=prep(window,args)
		if(len(thisWindowSeeds))>0:
			for localSeed in thisWindowSeeds:
				seeds.append((x+localSeed[1],y+localSeed[0]))
			#print("Current number of seeds "+str(len(seeds)))

	if args["dem"] is not None:
		dem = cv2.imread(args["dem"],0)
		if dem is None:raise Exception("no DEM at "+str(args["dem"]))
		seeds=refineWithDEM(dem,seeds)

	maskImage=255*np.ones((thresh.shape[0],thresh.shape[1],1),dtype=np.uint8)
	circleSize=40

	# create mask image
	for seed in seeds:
		# create binary result image
		cv2.circle(maskImage, seed, circleSize, 0, -1)

	if args["binOut"] is not None: out_binary = args["binOut"]
	else: out_binary = "SafonovaWS"+str(size)+".jpg"

	cv2.imwrite(out_binary, maskImage)



if __name__ == '__main__':

	main(sys.argv)
