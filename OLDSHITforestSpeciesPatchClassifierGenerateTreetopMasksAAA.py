import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import cv2
import numpy as np
import sys
from fastai import *
from fastai.vision import *
from treeTopPatcher import listFromBinary,getSquare

def main(argv):

    #Before we begin, check installation
    print("Checking if cuda is available "+str(torch.cuda.is_available()))
    print("Checking if cuddn is available "+str(torch.backends.cudnn.enabled))

    patchSize=100

    #Parameters
    PATH = pathlib.Path(argv[1])

    #print("Data path in "+str(PATH))
    # To see what files are in the PATH folder
    #print(PATH.ls())

    # argv[2] contains the model file name, where the weights of the network are stored
    # if we are training, it will be the file that we will use to SAVE the model
    # if we are not training, we will LOAD a model with this name, must match the architecture
    modelFileName=argv[2]

    # Argv[3] Define architecture, tested resnet and vgg, there are more
    if argv[3]=="res": arch=models.resnet50
    elif argv[3]=="vgg":arch=models.vgg16_bn
    elif argv[3]=="squ":arch=models.squeezenet1_0
    elif argv[3]=="dense":arch=models.densenet121
    elif argv[3]=="alex":arch=models.alexnet
    elif argv[3]=="wRes":arch=models.wrn_22

    else: raise Exception("fastai forest Species classifier, unrecognised architecture")
    # many more models are possible
    #https://docs.fast.ai/vision.models.html

    # argv[4] contains a code on what to do, if it contains "t" it will train
    # if it contains "p" it will predict categories for new images
    # Are we training the model or just loading one?
    training="t" in argv[4]

    # are we going to predict images not in the training/testing set?
    predict="p" in argv[4]

    offset=0
    if training:
        numEpochs=int(argv[5])
        lr=float(argv[6])
        offset=2
        print("TESTING ARCHITECTURE "+argv[3]+" with learning rate "+str(lr))
    if predict:
        #the following parameter (either 5 or 7 depending on whether training is True)
        # contains a file with a list of the names of the images to predict
        predictFileName=argv[5+offset]
        mosaic=cv2.imread(argv[6+offset],cv2.IMREAD_COLOR)
        if mosaic is None: raise Exception("forestSpeciesPatchClassifier, no mosaic read")

    bs=64 # size of batch, if you run out of memory switch to 16 (slower)

    # Get data file names, each image name will be extracted with REGULAR expressions
    fnames = get_image_files(PATH)
    pat = r'/SP([^/]+)PATCH\S*\d+.jpg$'
    # Regular expression for file name!!! Some references in case you are interested
    #https://docs.python.org/3/howto/regex.html
    #https://www.guru99.com/python-regular-expressions-complete-tutorial.html

    np.random.seed(2)

    # Read data from the path and using the regular expression defined in "pat"
    data = ImageDataBunch.from_name_re(PATH, fnames, pat,
    ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)
    print("Printing classes found! ")
    print(data.classes)

    learn = cnn_learner(data, arch, metrics=error_rate)

    if training:
        learn.fit_one_cycle(numEpochs, slice(lr))
        learn.save(modelFileName)
        interp = ClassificationInterpretation.from_learner(learn)
        conf=interp.confusion_matrix()
        print(conf)
        #valid=data.valid_ds.x.items[:]
        #print("validation data "+str(valid))
        # Make confusion matrix out of these!!!!
#        for x in valid:
#            aux=x.split("/")[-1].split(".")[0]
            # now copy the correct labels on to the validation dictionary
#            self.validFileDict[aux]=self.fileDict[aux]
            #print(self.validFileDict[aux])


    else: # Not training, load model
        learn.load(modelFileName)

    print("model trained or loaded, now compute predictions? "+str(predict))
    if(predict):

        alltops=listFromBinary(predictFileName) # now the prediction file is a binary file with treeTops marked
        predClassDict={}
        for cent in alltops:
            # classify t
            # add to a list of  its class

            try:
                # opencv works with inverted coords, so we have to invert ours.
                square = getSquare(patchSize, (cent[1],cent[0]), mosaic)

                #this is very dirty, can't find a better way!
                cv2.imwrite("./tempImage.jpg",square)
                img=open_image("./tempImage.jpg")
                pred_class,pred_idx,outputs =learn.predict(img)
                #print("predicted "+str(pred_class))
                if pred_class not in predClassDict:predClassDict[pred_class]=[(int(cent[0]),int(cent[1]))]
                else:predClassDict[pred_class].append((int(cent[0]),int(cent[1])))
                    #if label in str(pred_class).split(";"): maskW=np.add(maskW,maskOfOnes)
            except AssertionError as error:
                print(error)

        # now paint every point in a file
        for k,v in predClassDict.items():
            print("class "+str(k)+" \n with list of items "+str(v))

            maskImage=255*np.ones((mosaic.shape[0],mosaic.shape[1],1),dtype=np.uint8)
            circleSize=40

            for seed in v:
                cv2.circle(maskImage, seed, circleSize, 0, -1)
            print("saving "+"./"+str(k)+"tops.jpg")
            cv2.imwrite("./"+str(k)+"tops.jpg",maskImage)


if __name__ == '__main__':
    main(sys.argv)
