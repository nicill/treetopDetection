
methodString="0 0Ref60 0Ref75 1 3 3Ref60 3Ref75 5Ref50 5Ref60 5Ref75"

for method in $methodString;
do
	 python processOutput.py /media/yago/workDrive/Experiments/forests/ZaoProcessing/outputClustering/$method"percent.txt" /media/yago/workDrive/Experiments/forests/ZaoProcessing/outputClustering/$method"pointDiff.txt" cluster
done




