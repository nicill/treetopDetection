import sys

def main(argv):

    if len(argv)>3 and argv[3]=="cluster":clusteringOutput=True
    else: clusteringOutput=False

    with open(argv[1]) as f1, open(argv[2]) as f2:
        count=0

        for x, y in zip(f1, f2):
            if count==0:
                print("\t\t\t\t Perc 1 2 3 4 5 6 7 8 9 PointDif 1 2 3 4 5 6 7 8 9")
                count=1
            else:
                secondPart=""
                if not clusteringOutput:
                    for a in y.strip().split(" ")[6:]:secondPart+=" "+a
                else:
                    for a in y.strip().split(" ")[1:]:secondPart+=" "+a

                print("{0}\t{1}".format(x.strip(), secondPart))


if __name__ == '__main__':
    main(sys.argv)
