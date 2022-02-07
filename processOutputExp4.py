import sys

def isInteresting(line):
    listOfInterestingThings=["models","all sites","training site 3 done","sensitivity","exp4UNF"]
    for x in listOfInterestingThings:
        if x in line:return True

    return False or line[:2]=="9 "

def main(argv):
    classes=["deciduous","healthyfir","sickfir"]

    mode=int(argv[1])
    resDict={}
    numClasses=3
    numSites=3

    targetClass=argv[3]
    critID=int(argv[4])

    if mode==1:# training individual sites, without all sites
        currentArch="noArch"
        modelName="noName"
        with open(argv[2]) as f1:

            for x in f1:
                if isInteresting(x):
                    #print(x.strip())
                    if "models" in x:
                        print(x)
                        modelName=x.strip().split(" ")[2].split("L")[0].split("l")[1]+"LR"+x.strip().split(" ")[2].split("R")[1].split("e")[0]
                        resDict[modelName]=[]
                        #print(modelName)
                    if "training site 3 done" in x:
                        change=True
                    elif targetClass+" sensitivity" in x:
                        print(x)
                        start=4
                        allCriteria=(float(x.strip().split(" ")[start][:-1]),float(x.strip().split(" ")[start+3][:-1]),float(x.strip().split(" ")[start+6][:-1]))
                        resDict[modelName].append(allCriteria[critID])

        print(resDict)
        #print("\n")


        # HERE WE SHOULD OUTPUT AVERAGE PER CLASS
        print("weighted average between all sites ")
        bestValue=-1
        bestModel=(0,0,0)
        weights=[45,44,80]
        total=sum(weights)
        for k,v in resDict.items():
            #print(v)
            try:
                for site in range(numSites):
                    weightAv=(weights[0]*v[0]+weights[1]*v[1]+weights[2]*v[2])/total
                    print(str(k)+" "+str(weightAv))

                if bestValue==-1 or ( weightAv >bestValue):
                    bestValue=weightAv
                    bestModel=(k,weightAv,v)

            except Exception as E:
                print("problem when checking all sites "+str(E))

        print(bestValue)
        print(bestModel)


    if mode==2:# take file with just one site and extract error rate for every learning rate.
        currentModel="NOMODEL"
        augm="NOIDEAAUGM"
        with open(argv[2]) as f1:
            for x in f1:
                if isInteresting(x):
                    #print(x.strip())
                    if "models" in x:
                        print(x)
                        modelName=augm+x.strip().split(" ")[2].split("L")[0].split("l")[1]+"LR"+x.strip().split(" ")[2].split("R")[1].split("e")[0]
                        if not modelName in resDict: resDict[modelName]=[]
                        currentModel=modelName
                    #elif x[:2]=="9 ":
                        #print(x.strip().split(" "))
                    #    resDict[currentModel].append(float(x.strip().split(" ")[-5][:-1]))
                    if "exp4UNF" in x:
                        print(x)
                        augm="augm"+str(x.strip().split("F")[1].split("a")[1][6:])
                        print(augm)
                    elif "sickfir sensitivity" in x:
                        start=0
                        print(x)
                        # sickfir accuracy
                        #resDict[modelName].append(float(x.strip().split(" ")[4][:-1]))
                        resDict[modelName].append((float(x.strip().split(" ")[4][:-1]),float(x.strip().split(" ")[7][:-1]),float(x.strip().split(" ")[10][:-1])))
#        print(resDict)

        # make average over the 4 sites
        for k,v in resDict.items():
            avs=[0,0,0]
            outString=str(k)
            for x in v:#loop over tuples
                for i in range(3):
                    avs[i]+=x[i]
            for i in range(3):
                outString+=" "+str(avs[i]/len(v))
            print(outString)


#        for k,v in resDict.items():
#            try:arch=str(k).split("h")[1].split("L")[0]
#            except Exception as E: arch="NOARCH"
#            try:lRate=str(k).split("R")[1]
#            except Exception as E: lRate="NOLR"
#            try:er=str(v[0])
#            except Exception as E: er="NOER"

#            print(arch+" "+lRate+" "+er)

#        bestSickfirSens=-1
#    bestModel=(0,0)

if __name__ == '__main__':
    main(sys.argv)
