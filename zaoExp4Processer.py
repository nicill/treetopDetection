import sys
import cv2
import crownSegmenterEvaluator as cse

def howManyCaughtDetection(argv):
    # argv[1] contains the prefix of the ground truth
    gtPrefix=argv[1]
    #read images and translate them into point lists
    ROIFileName=gtPrefix+"ROI.jpg"

    # argv[2] contains the prefix of all the predicted points
    predPrefix=argv[2]
    predImageName=predPrefix+"Best.jpg"
    allTopsPredicted=cse.listFromBinary(predImageName,ROIFileName)

    epsilon=int(argv[3])

    #the rest of argv contains the list of classes to be read


    classAndPointsList=[]
    for i in range(4,len(argv)):
        #print("reading"+argv[i])
        #cv2.imread()
        gtImageName=gtPrefix+argv[i]+"tops.jpg"
        #gtImage=cv2.imread(gtImageName,cv2.IMREAD_GRAYSCALE)
        #if gtImage is None: raise Exception(gtImageName+" not read!!!!!!!!!")
        gtList=cse.listFromBinary(gtImageName,ROIFileName)

        classAndPointsList.append((argv[i],gtList))
        print("For class "+argv[i]+" number of real points "+str(len(gtList)))
        #print("For class "+argv[i]+" num real "+str(len(gtList))+" num predicted "+str(len(predList)))



    #print(classAndPointsList)
    #print("\n\n"+str(allTops))
    for i in range(4,len(argv)):
        #print(argv[i]+" points with possible matching before classification "+str(format(cse.matchedPercentage(classAndPointsList[i-4][1],allTopsPredicted,epsilon),'.2f')))
        print(argv[i]+" points with possible matching before classification "+str(cse.matchedPercentage(classAndPointsList[i-4][1],allTopsPredicted,epsilon)))
        #print(argv[i]+" points with matching after classification "+str(format(cse.matchedPercentage(classAndPointsList[i-4][2],classAndPointsList[i-4][1],epsilon),'.2f')))



def evaluateTreeTopPredictionWithClasses(argv):
    # argv[1] contains the prefix of the ground truth
    gtPrefix=argv[1]

    # argv[2] contains the prefix of the predicted points
    predPrefix=argv[2]

    epsilon=int(argv[3])

    #the rest of argv contains the list of classes to be read

    #read images and translate them into point lists
    ROIFileName=gtPrefix+"ROI.jpg"

    classAndPointsList=[]
    for i in range(4,len(argv)):
        #print("reading"+argv[i])
        #cv2.imread()
        gtImageName=gtPrefix+argv[i]+"tops.jpg"
        #gtImage=cv2.imread(gtImageName,cv2.IMREAD_GRAYSCALE)
        #if gtImage is None: raise Exception(gtImageName+" not read!!!!!!!!!")
        gtList=cse.listFromBinary(gtImageName,ROIFileName)

        predImageName=predPrefix+argv[i]+"tops.jpg"
        #predImage=cv2.imread(predImageName,cv2.IMREAD_GRAYSCALE)
        #if predImage is None: raise Exception(predImageName+" not read!!!!!!!!!")
        predList=cse.listFromBinary(predImageName,ROIFileName)

        classAndPointsList.append((argv[i],gtList,predList))
        #print("For class "+argv[i]+" number of real points "+str(len(gtList)))
        #print("For class "+argv[i]+" num real "+str(len(gtList))+" num predicted "+str(len(predList)))


    # find out the list of all tops
    allTopsPredicted=[]
    for i in range(4,len(argv)):
        for x in classAndPointsList[i-4][2]:allTopsPredicted.append(x)

    #print("total "+str(len(allTopsPredicted)))

    #print(classAndPointsList)
    #print("\n\n"+str(allTops))
    outString=""
    for i in range(4,len(argv)):
        #print(argv[i]+" points with possible matching before classification "+str(cse.matchedPercentage(classAndPointsList[i-4][1],allTopsPredicted,epsilon,1)))
        #print(argv[i]+" points with matching after classification "+str(format(cse.matchedPercentage(classAndPointsList[i-4][2],classAndPointsList[i-4][1],epsilon,1),'.2f')))
        outString+=" "+str(len(classAndPointsList[i-4][1]))+" "+str(cse.matchedPercentage(classAndPointsList[i-4][1],allTopsPredicted,epsilon,1))+" "+str(cse.matchedPercentage(classAndPointsList[i-4][1],classAndPointsList[i-4][2],epsilon,1))+" "+str(len(classAndPointsList[i-4][2]))+" "+str(cse.matchedPercentage(classAndPointsList[i-4][2],classAndPointsList[i-4][1],epsilon,1))
    return outString
if __name__ == '__main__':
    print(evaluateTreeTopPredictionWithClasses(sys.argv),end="")
    #howManyCaughtDetection(sys.argv)
