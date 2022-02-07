import numpy as np
import cv2
import sys


def main(argv):

    if (int(argv[1]))==0: # read a mask image and a ROI and erase the masked parts from the roi

        roi = cv2.imread(argv[2],0)
        if roi is None:raise Exception("imageUtils no ROI at "+str(argv[2]))

        treeMask = cv2.imread(argv[3],0)
        if treeMask is None:raise Exception("imageUtils no maskTops at "+str(argv[3]))

        roi[treeMask==0]=255

        cv2.imwrite(argv[2],roi)

    else:
        raise Exception("demUtils, wrong code")

if __name__ == '__main__':
    main(sys.argv)
