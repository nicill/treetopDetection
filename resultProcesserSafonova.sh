
methodString="testSAFO"

for method in $methodString;
do
	 python processOutput.py /media/yago/workDrive/Experiments/forests/ZaoProcessing/outputSafonova/$method"percent.txt" /media/yago/workDrive/Experiments/forests/ZaoProcessing/outputSafonova/$method"pointDiff.txt" cluster
done




