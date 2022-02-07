import sys

def computeBestDict(th,filePrefix):
#argv[1] matching percentage file
#argv[2] pointDif file
#argv[3] threshold for point diff
#argv[4] euc file
#argv[5] haus file file
#argv[6] eucMatched file
#argv[7] repeated file


    bestDict={}
    numMosaics=9
    offset=6
    threshold=int(th)
    #print("THRESHOLD!!! "+str(threshold))
    for i in range(1,numMosaics+1):
        bestDict[i]=(0,0,0,0,0,0,0,"")
    #print(bestDict)

    filePercent=filePrefix+"percent.txt"
    fileDiff=filePrefix+"pointDiff.txt"
    fileEuc=filePrefix+"euc.txt"
    fileHaus=filePrefix+"haus.txt"
    fileEucMatch=filePrefix+"eucMatched.txt"
    fileRepe=filePrefix+"repeated.txt"

    try:
        with open(filePercent) as f1, open(fileDiff) as f2,open(fileEuc) as f3,open(fileHaus) as f5,open(fileEucMatch) as f4,open(fileRepe) as f6:
            count=0

            for x,y,z,t,u,v in zip(f1,f2,f3,f4,f5,f6):
                if count==0:count+=1
                else:
                    # retrieve (match perc and rest)
                    percentSplit=x.strip().split(" ")
                    pointDiffSplit=y.strip().split(" ")
                    eucSplit=z.strip().split(" ")
                    eucMSplit=t.strip().split(" ")
                    hausSplit=u.strip().split(" ")
                    repeSplit=v.strip().split(" ")



                    for i in range(1,numMosaics+1):
                        try:
                            pointDiff=float(pointDiffSplit[offset + i-1])

                            if abs(pointDiff)<threshold:
                                match=float(percentSplit[offset + i-1 ])

                                if match>bestDict[i][0]:
                                    euc=float(eucSplit[offset + i-1])
                                    eucM=float(eucMSplit[i-1])
                                    haus=float(hausSplit[offset + i-1])
                                    repe=float(repeSplit[i-1])
                                    outputString=filePrefix
                                    for st in percentSplit[:offset + i-1]:outputString+=st
                                    bestDict[i]=(match,abs(pointDiff),pointDiff,euc,eucM,haus,repe,outputString)
                                    #print("helou "+str(bestDict))
                                #print(str(i)+"match "+str(match))
                                #print("pointDiff "+str(pointDiff))
                        except:
                            #print("AEXCEPTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            continue


                    #for a in y.strip().split(" ")[offset:]:secondPart+=" "+a

    except Exception as e:
        #print("EXCEPTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(e)

    #print("END!!!!!!!!!!!!!!!!!!!!!!!!"+str(bestDict))
    return bestDict

def main(argv):
    th=argv[1]
    absoluteBestDict={}
    for x in range(2,len(argv)):
        #print("computing best dict for "+str(argv[x]))
        currentbestDict=computeBestDict(th,argv[x])
        #print("CURRENT "+str(currentbestDict))
        for k,v in currentbestDict.items():
            if k not in absoluteBestDict or absoluteBestDict[k][0]<v[0] :
                #print("IMPROVING! "+str(k))
                absoluteBestDict[k]=v

    #print("\n\nAbsolute "+str(absoluteBestDict))
    for k,v in absoluteBestDict.items():
        print(str(k)+" "+str(v[0])+" "+str(v[1])+" "+str(v[2])+" "+str(v[3])+" "+str(v[4])+" "+str(v[5])+" "+str(v[6])+" "+str(v[7]))


if __name__ == '__main__':
    main(sys.argv)
