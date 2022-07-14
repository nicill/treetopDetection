# treetopDetection

This is a repository of the code for tree top detection developed by [Yago Diez](http://yagodiez.com/) in collaboration with several members of the Forestry group at Yamagata University lead by [Larry Lopez](https://www.tr.yamagata-u.ac.jp/~larry/profile.html). This includes Ha Trang Nguyen, Herve Gonrou Dobou, Leonardo Silvestre and Sergi Garcia Riera).

Version of this code have been used in [this paper](https://www.mdpi.com/2072-4292/13/2/260) and [this other paper](https://www.mdpi.com/2079-3197/10/6/90)

## Installation Instructions

This project has been tested in a linux Ubuntu environment and it is strongly encouraged that it is run in linux environments..

The main code files are written in Python with opencv, scikit-learn and numpy as their main libraries.

TODO: Requirements File!

Test scripts written in bash are used to set up the experiments (see Testing for some details)


## Project Structure

The main file is the "sliding window" file which uses a connected component approach to detect tree tops. The "main" function reads an elevation map representing a DEM or CHM in "tif" format with float values and uses the "sliding window" function to cut the DEM into windows (at this moment, with 20% overlap). The size of the window that is used in encoded in the "size" argument (wSizes in the bash file)

In each window, the "processWindow" function is called. This function first calls the "binarize" function that a) Erases the lower points in each window to try to separate trees b) Erodes the resulting image with the same goal. After this, the "processWindow" function calls the "findTops" function. At this moment the 20% of lower pixels are erased and a 2-pixel square element is used with 3 iterations to erode the image (default parameters in the binarize function).

The "findTops" function finds treetops in every window by considering increasingly wide bands of altitudes (first only the top pixels are considered and more an more are considered in every step). The width of the band that is added at each step is encoded in the "topStep" argument (minDists in the bash file)

In each band, the visible connected components are considered and in newly appearing components new tops are assigned to the highest point of any new component that is "big enough". This is encoded by the "minPixTop" argument (thresholds in the bash file).

There is also a "refineRadius" argument (both in python and bash) that is used to eliminate tops that are two close to each other. This happens at every iteration of the connected component algorithm as well as after all the windows have been processed.


## Testing

I recommend testing using the bash scripts, but you can use the calls in them to design your own tests. An example is:

bash crownSegmTestEPSHerve.sh CODEFOLDER OUTPUTFOLDER DATAFOLDER ce prefix FinalRefinemenr RefineRadius

For example in my case this is:

bash crownSegmTestEPSHerve.sh /home/x/Experiments/treetopDetection/ /home/x/Experiments/treetopDetection/output/ /home/x/Experiments/treetopDetection/Sergi/ ce newref5 no 5

It is important that all folders (including the output folder) must exist before calling the script.

in line 25 you will find the epsilon values considered to assess the quality of matching.

in lines 58-60 you will find the values considered for the main parameters.

## Interpretation of Results

The code produces a series of outputimages depicting the location of the predicted points. These are stored in the data folder and the file names contain the Dem that they correspond to and the parameters that were used to create them. 

The code also produces several "criterion" files according to the definitions found in file "crownSegmenterEvaluator.py". As the data is pretty large, at this moment only two criteria are considered:

1) "percent", percentage of real points that are matched (distance < than the each of the epsilon values defined in line 25 for each dem)

2) pointDiff, percentual difference between the number of predicted and existing points. +7 means that there are 7% more real points than predicted, 0% means that the number of predicted and real points is exactly the same. -15% means that there are 15% more predicted points than real points.
