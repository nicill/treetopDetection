import matplotlib.pyplot as plt
import csv
import sys

def main(argv):

    x = []
    y = []

    #print("opening "+str(argv[1]))
    yLabel=""
    xLabel=[]
    #plt.figure(dpi=1000)
    with open(argv[1],'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            #first row, find out number of categories
            if yLabel=="":
                yLabel=row[0]
                for i in range(1,len(row)):
                    x.append([])
                    xLabel.append(row[i])
            else:
                #print(row)
                y.append(float(row[0]))
                for i in range(len(row)-1):
                    x[i].append(float(row[i+1]))

    #print(x)
    for i in range(len(x)):
        #print(xLabel[i])
        if xLabel[i][:3]=="Unf":plt.plot(y,x[i], label=xLabel[i], linewidth=3)
        else:plt.plot(y,x[i],"--", label=xLabel[i], linewidth=3)

    #code=int(argv[2])
    code=0

    labelsLong=["Error Rate"]
    labelsShort=["ER"]

    plt.xlabel(yLabel)
    plt.ylabel(labelsShort[code]+" %")
    plt.xscale('log')
    plt.title(labelsLong[code]+' %')
    plt.legend(loc=9,ncol=2,prop={"size":20})
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.030,0.12))

    plt.show()


    #plt.savefig(argv[2])

if __name__ == '__main__':
    main(sys.argv)
