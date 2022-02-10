#ALGORITHM TO WORK AS SCRIPT
import cv2
import numpy as np
import time
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import sys


def execute_algorithm(image,min_hieght,result_file,result_image,size,its):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    time_start = time.perf_counter()
    ret,thresh = cv2.threshold(image,min_hieght,255,cv2.THRESH_TOZERO)
    im_erode = cv2.erode(thresh,kernel,iterations=its)

    norm_original = cv2.normalize(image,dst=None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    normalized = cv2.normalize(im_erode,dst=None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    im_show = cv2.cvtColor(norm_original,cv2.COLOR_GRAY2RGBA)
    blurred = cv2.GaussianBlur(im_erode,(size,size),0)

    im_dilate = cv2.dilate(blurred,kernel,iterations=2)

    im_comp = cv2.compare(blurred,im_dilate,cv2.CMP_GE)
    im_erode = cv2.erode(blurred,kernel,iterations=2*its)
    im_plateu = cv2.compare(blurred,im_erode,cv2.CMP_GT)
    im_and = cv2.bitwise_and(im_comp,im_plateu)

    #cv2.imshow("IM_AND",im_and)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    output = cv2.connectedComponentsWithStats(im_and,4,cv2.CV_32S)
    (numLabels,labels,stats,centroids) = output
    time_end = time.perf_counter()
    mask = np.zeros(image.shape,dtype='uint8')
    elapsed_time = time_end-time_start
    print("Elapsed time (seconds):",elapsed_time)
    print("Trees found",numLabels)
    lines = []
    for cent in centroids:
        cv2.circle(im_show,(int(cent[0]),int(cent[1])),5,(0,0,255))
        mask[int(cent[1]),int(cent[0])] = 255
        t_h = image[int(cent[1]),int(cent[0])]
        lines.append((str(cent[0]),str(cent[1]),str(t_h)))

    with open(result_file,'w+') as ofil:
        for lin in lines:
            ofil.write("%s,%s,%s \n"%(lin[0],lin[1],lin[2]))
    # CREATE BINARY MASK WITH TREETOPS
    ret,mask_inv = cv2.threshold(mask,0,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite(result_image,mask_inv)

    #RESIZE IMAGE
    # scale_percent = 60 # percent of original size
    # width = 1800
    # height = 900
    # width = int(im_show.shape[1] * scale_percent / 100)
    # height = int(im_show.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # resized = cv2.resize(im_show,dim,interpolation=cv2.INTER_AREA)
    # return resized
    #return im_show


if __name__ == '__main__':

    path_image = sys.argv[1]
    result_image = sys.argv[2]
    result_file = sys.argv[3] # set name of results file
    min_tree_height = float(sys.argv[4]) #set minimun tree height
    kernelSize=int(sys.argv[5])
    morphIterations=int(sys.argv[6])
    #If needed can copy lines 66,68,69 and add more images, needs to change the name of the variables
    original_image = cv2.imread(path_image,cv2.IMREAD_UNCHANGED)

    execute_algorithm(original_image,min_tree_height,result_file,result_image,kernelSize,morphIterations)

    #shows images with opencv viewer
    #cv2.imshow("Result Image",treetops)


    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
