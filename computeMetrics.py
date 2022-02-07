def computeMetrics(list,classes=None):
    total=0
    for l in list:
        for n in l:
            total+=n

    #compute sensitivity and specificity
    for i in range(len(list)):
        TP=list[i][i]
        P=sum(list[i])
        if P==0:
            sen=0
        else:
            sen=TP/P

        columnSum=0
        for c in range(len(list)):
            columnSum+=list[c][i]

        TN=total-sum(list[i])-columnSum+TP
        N=total-P
        spe=TN/N

        acc=(TP+TN)/total

        print("sensitivity : "+str(sen)+", specificity : "+str(spe)+", accuracy : "+str(acc)+"\n")



def main():
    #confusion matrix,  list of list
    List=[[267,4,1],[1,821,3],[0,19,20]]

    computeMetrics(List)

if __name__ == '__main__':
    main()
