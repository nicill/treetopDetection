import cv2
#import numpy as np
import sys
from imgaug import augmenters as iaa

import random

def augment(image,code,outputFile,verbose=True):
    if code==0:
        if verbose: print("Doing Data augmentation 0 (H fip) to image "+outputFile)
        #image_aug = iaa.Fliplr(1.0)(images=image)
        image_aug = iaa.Rot90(1)(image=image)
        cv2.imwrite(outputFile,image_aug)
    elif code==1:
        if verbose: print("Doing Data augmentation 1 (V flip) to image "+outputFile)
        image_aug = iaa.Flipud(1.0)(image=image)
        cv2.imwrite(outputFile,image_aug)
    elif code==2:
        if verbose: print("Doing Data augmentation 2 (Gaussian Blur) to image "+outputFile)
        image_aug = iaa.GaussianBlur(sigma=(0, 0.5))(image=image)
        cv2.imwrite(outputFile,image_aug)
    elif code==3:
        if verbose: print("Doing Data augmentation 3 (rotation) to image "+outputFile)
        angle=random.randint(0,45)
        rotate = iaa.Affine(rotate=(-angle, angle))
        image_aug = rotate(image=image)
        cv2.imwrite(outputFile,image_aug)
    elif code==4:
        if verbose: print("Doing Data augmentation 4 (elastic) to image "+outputFile)
        image_aug = iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.1)(image=image)
        cv2.imwrite(outputFile,image_aug)
    elif code==5:
        if verbose: print("Doing Data augmentation 5 (contrast) to image "+outputFile)
        image_aug=iaa.LinearContrast((0.75, 1.5))(image=image)
        cv2.imwrite(outputFile,image_aug)
    else:
        print("Doing some other Data augmentation to image "+outputFile)
        #https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html

def isPatchToDownsample(classList, classesToDownsample):
    retList=[]
    for x in classList:
        if x not in classesToDownsample: retList.append(x)
    return len(retList)==0


def processCSVFile(csvFile,outFile,imageFolder,outputImageFolder,classesToAugment, classesToDownsample, verbose=False):

    f = open(csvFile, "r")
    f2 = open(outFile, "w")

    if imageFolder[-1]!="/":imageFolder=imageFolder+"/"
    if outputImageFolder[-1]!="/":outputImageFolder=outputImageFolder+"/"

    # Augmentation factors
    augmentAlone=50
    augmentMixed=20

    #number Of available augmentations
    numAugm=6

    #for every line
    countTotal=0
    countPresent=0
    countAlone=0
    exceptionCount=0
    for line in f:
        #print(line)
        imageName=line.split(",")[0]
        if imageName!="image":
            countTotal+=1
            #load image
            imageFile=imageFolder+imageName+".jpg"
            currentImage=cv2.imread(imageFile)
            try:
                if currentImage is None:
                    raise Exception("multiLabelDataAugmenter:processCSVFile, image "+str(imageFile)+" was not found")
                # get its list of labels
                labelList=line.strip().split(",")[1].split(" ")
                labelString=labelList[0]
                for x in labelList[1:]:labelString=labelString+" "+x

                # also, write the news output file
                donwsamplePerct=80
                if isPatchToDownsample(labelList,classesToDownsample) and (random.randint(0,100)>donwsamplePerct):f2.write(line)

                for x in classesToAugment:
                    if x in labelList:
                        countPresent+=1

                        if len(labelList)==1:
                            countAlone+=1
                            # Alone, do the augmentations marked by augmentAlone
                            for i in range(augmentAlone):
                                augment(currentImage,i%numAugm,outputImageFolder+imageName+"AA"+str(i)+".jpg",verbose)
                                #also update File
                                f2.write(imageName+"AA"+str(i)+","+labelString+"\n")
                        else:
                            #Present not alone, do the augmentations marked by augmentMixed
                            for i in range(augmentMixed):
                                augment(currentImage,i%numAugm,outputImageFolder+imageName+"AM"+str(i)+".jpg",verbose)
                                #also update File
                                f2.write(imageName+"AM"+str(i)+","+labelString+"\n")
            except Exception as e: #only for debugging purposes
                print(e)
                exceptionCount+=1

    print("Exceptions percentage "+str(100*(exceptionCount/countTotal)))
    #print("Present in "+str(countPresent)+" "+str(100*(countPresent/countTotal)))
    #print("Alone in "+str(countAlone)+" "+str(100*(countAlone/countTotal)))
    f.close()
    f2.close()

def statsCSVFile(csvFile):
    allLabels=[]
    posDict={}
    #First, read classes and keep track of their position in the list
    f = open(csvFile, "r")
    for line in f:
        imageName=line.split(",")[0]
        if imageName!="image":
            labelList=line.strip().split(",")[1].split(" ")
            for x in labelList:
                if x not in allLabels:
                    allLabels.append(x)
                    posDict[x]=len(allLabels)-1
    print(allLabels)
    print(posDict)
    f.close()

    #Now compute statistics
    f = open(csvFile, "r")
    totalPatches=0
    listOfSingleOccurrences=[0]*len(allLabels)
    listOfSingletons=[0]*len(allLabels)
    couplesDict={}

    for line in f:
        #print(line)
        imageName=line.split(",")[0]
        if imageName!="image":
            totalPatches+=1
            labelList=line.strip().split(",")[1].split(" ")
            for x in labelList:
                listOfSingleOccurrences[posDict[x]]+=1
            #also pairs
            for i in range(len(labelList)):
                for j in range(i+1,len(labelList)):
                    a=labelList[i]
                    b=labelList[j]
                    if (a,b) in couplesDict:couplesDict[(a,b)]+=1
                    else:couplesDict[(a,b)]=1
            #also patches with only one class
            if len(labelList)==1:
                for x in labelList:
                    listOfSingletons[posDict[x]]+=1

    print("total Patches "+str(totalPatches))
    # normalize pairs
    for k,v in couplesDict.items():
        newV=100*v/totalPatches
        couplesDict[k]=newV
    print(listOfSingleOccurrences)
    for i in range(len(allLabels)):
        print(allLabels[i]+" appears "+str(100*(listOfSingleOccurrences[i]/totalPatches))+" percent of times ")
    print(couplesDict)
    print("list of singletons "+str(listOfSingletons))

def main(argv):
    csvF=argv[1]
    outputCsv=argv[2]
    imageFolder=argv[3]
    outputImageFolder=argv[4]

    classesToAugment=["blueberry"]
    classesToDownsample=["soil"]
    statsCSVFile(csvF)
    #processCSVFile(csvF,outputCsv,imageFolder,outputImageFolder,classesToAugment,classesToDownsample,False)
    #statsCSVFile(outputCsv)

if __name__== "__main__":
  main(sys.argv)
