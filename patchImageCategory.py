import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

def borderPoint(image, point):
    margin=100
    top1=image.shape[0]
    top2=image.shape[1]

    return point[0]<margin or (top1-point[0])<margin or point[1]<margin or (top2-point[1])<margin

def listFromBinary(fileName):

    #open filename
    im=cv2.imread(fileName,cv2.IMREAD_GRAYSCALE)
    #print(im.shape)
    if im is None:
        raise Exception("tree top image is not given")
        return []
    else:
        mask = cv2.threshold(255-im, 40, 255, cv2.THRESH_BINARY)[1]

        #compute connected components
        numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)
        #print("crownSegmenterEvaluator, found "+str(numLabels)+" "+str(len(centroids))+" points for file "+fileName)

        #im2 = 255 * np. ones(shape=[im.shape[0], im.shape[1], 1], dtype=np. uint8)

        #print(" listFromBinary, found  "+str(len(centroids)))
        #print(centroids)

        newCentroids=[]
        for c in centroids:
            if not borderPoint(im,c): newCentroids.append(c)
        print("refind "+str(len(newCentroids))+" points for file "+fileName)
        return newCentroids[1:]


def getSquare(w_size, p, img):
	height, width, channels = img.shape

	#isInside = (int(p[0])-w_size//2) >= 0 and (int(p[0])+w_size//2) < width and (int(p[1])-w_size//2) >= 0 and (int(p[1])+w_size//2) < height
	isInside = (int(p[1])-w_size//2) >= 0 and (int(p[1])+w_size//2) < width and (int(p[0])-w_size//2) >= 0 and (int(p[0])+w_size//2) < height

	assert isInside, "The required window is out of bounds of the input image "+str(p)

	return img[int(p[0])-w_size//2:int(p[0])+w_size//2, int(p[1])-w_size//2:int(p[1])+w_size//2]

def createPatchImage(mosaic, center ,patchSize):

    patchimage=getSquare(patchSize, (center[1],center[0]), mosaic)
    #print(int(center[0]),int(center[1]))
    #cv2.imwrite("patchImage.jpg", patchimage)
    im_rgb = cv2.cvtColor(patchimage, cv2.COLOR_BGR2RGB)
    return im_rgb

def binaryimageAndWhitepixel(grayImage):
    #using threshold, change gray scale image to binary image
    #count the number of white pixels, return category

    binaryImage = cv2.threshold(grayImage, 170, 255, cv2.THRESH_BINARY)[1]
    whitepixel=countWhitepercentage(binaryImage)
    return binaryImage, whitepixel

def countWhitepercentage(image):
    height=image.shape[0]
    width=image.shape[1]

    totalpixel=height*width

    #conut the number of white pixels

    whitePix=cv2.countNonZero(image)
    #print((totalpixel,whitePix))
    whitePixPer=(whitePix/totalpixel)*100
    return round(whitePixPer, 2)

def writeTextfile(filename, listOfText):
    #write Some sentences in text file
    with open(filename, mode="w") as f:
        f.write("\n".join(listOfText))

def selsectPattern(event):
    #if the category of this patch is "C","S" or "RD", separate it by pattern
    #press the number key to save it as pattern numnber
    global pattern
    if event.key=="1":
        plt.close()
        pattern=event.key

    elif event.key=="2":
        plt.close()
        pattern=event.key

    elif event.key=="3":
        plt.close()
        pattern=event.key

def turnOffDisplay(event):
    if event.key=="p":
        plt.close()

def main(argv):
    mosaic=cv2.imread(argv[1], cv2.IMREAD_COLOR)
    if mosaic is None:
        raise Exception("mosaic is not given")

    treetop=argv[2]

    #create patch image from mosaic and tree top annotation image
    #find the coordinates of the tree top
    centroids=listFromBinary(treetop)
    #in each tree top, make patch images of indicated size
    patchSize=100

    #show RGB and Binary image for each patch
    #and create text file contain image pattern and categorize(healthy, colonized, sick, recently dead, dead)
    #arvg[3] receives the name of a text file
    filename=argv[3]
    #specify the number of the beginning of the tree in argv[4] and last number in argv[5]
    first=int(argv[4])
    last=int(argv[5])
    number=first

    lines=[]
    for cent in centroids[first-1:last]:
        patch=createPatchImage(mosaic, cent, patchSize)
        #change rgb image to grayscale image
        grayImg=cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        binaryImg, whitePixelPercent=binaryimageAndWhitepixel(grayImg)

        #determining category
        category="noCategory"
        if whitePixelPercent < 50:
            category="H"
        elif 50 <= whitePixelPercent < 60:
            category="C"
        elif 60 <= whitePixelPercent < 70:
            category="S"
        elif 70 <= whitePixelPercent < 80:
            category="RD"
        else:
            category="D"

        #if category is "H" or "D", pattern is 0
        #Otherwise, selsect a disease pattern
        global pattern
        pattern=0

        #show patch image on the screen
        fig=plt.figure()
        ax1=fig.add_subplot(1,2,1)
        ax1.set_title("RGB patch image")
        plt.imshow(patch)
        ax2=fig.add_subplot(1,2,2)
        ax2.set_title("Binary image")
        ax2.text(0, 120, "white pixel percentage : "+str(whitePixelPercent)+"%"+"\n"+"category : "+category)
        plt.imshow(binaryImg, cmap="gray")
        if category in ["C","S","RD"]:
            plt.connect("key_press_event", selsectPattern)
        else:
            plt.connect("key_press_event", turnOffDisplay)
        plt.show()

        #print("centroid number is "+str(number)+", coordinates is "+str((int(cent[0]),int(cent[1])))+", category : "+category+", pattern : "+str(pattern))
        line=str(number)+" "+category+" "+str(pattern)
        print(line)
        lines.append(line)
        number+=1

    #write patch number and categoris and pattern in text file
    writeTextfile(filename, lines)


if __name__ == '__main__':
    main(sys.argv)
