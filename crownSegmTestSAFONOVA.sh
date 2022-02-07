# /usr/bin/bash


#this needs to be called from inside a proper python environment!

#Example call
#./testScripts/crownSegmTest.sh /media/yago/workDrive/Experiments/forests/segmentation /media/yago/workDrive/Experiments/forests/segmentation/testScripts/ /media/yago/workDrive/Experiments/forests/segmentation/crown ce 

sourceDir=$1
outputDir=$2
dataPrefix=$3
whatToDo=$4
methodInput=$5
epsilon=$6

mosaicString="1 2 3 4 5 6 7 8 9"


hausFile=$outputDir/$methodInput"haus.txt"
eucFile=$outputDir/$methodInput"euc.txt"
percentFile=$outputDir/$methodInput"percent.txt"
reversePercentFile=$outputDir/$methodInput"reversePercent.txt"
pointDiffFile=$outputDir/$methodInput"pointDiff.txt"


# Codify the actions that the script will perform
compute=0
evaluate=0
shutdown=0
if [[ $whatToDo == *"c"* ]]; then
compute=1
fi
if [[ $whatToDo == *"e"* ]]; then
evaluate=1
fi
if [[ $whatToDo == *"s"* ]]; then
shutdown=1
fi

#echo "method $mosaicString " > $hausFile
#echo "method $mosaicString " > $eucFile
echo "method $mosaicString " > $percentFile
#echo "method $mosaicString " > $reversePercentFile
echo "method $mosaicString " > $pointDiffFile


	wSizes="500 1000 1500 2000 2500"
	thresholds="60 70 80 90 100 110 120 130 140 150 160 170 180"
	minPoints="100 250 500 1000 2500 3000 5000"
	for ws in $wSizes
	do
		for th in $thresholds
		do
			for md in $minPoints
			do

				paramStringVector+=("s$ws""th$th""mp$md")
				paramVector+=("-s $ws -th $th -mp $md")

			done
		done
	done

	l=${#paramVector[@]}

	for (( j=0 ; j<$l; j++ )) 
	do
 
		params=${paramVector[j]}
		paramString=${paramStringVector[j]}

		#echo "$method $params "

		echo -n "$method $params " >> $hausFile
		echo -n "$method $params " >> $eucFile
		echo -n "$method $params " >> $percentFile
		echo -n "$method $params " >> $reversePercentFile
		echo -n "$method $params " >> $pointDiffFile

		for i in $mosaicString
		do
				

			echo "doing method $method with mosaic $i and params $params "
			outputFileName=$dataPrefix"/IMAGE"$i"_binaryMethod"$method$paramString".jpg"
			roiMaskFileName=$dataPrefix"/Z"$i"ROI.jpg"

			#echo "outputFile $outputFileName " 
			
			if [[ $compute = 1 ]];then
		
				START=$(date +%s)
				echo "python $sourceDir/Safonova.py -d $dataPrefix"/Z"$i"nDEM.jpg" $params -o $outputFileName "			

				python $sourceDir/Safonova.py -i $dataPrefix"/Z"$i".jpg" -d $dataPrefix"/Z"$i"nDEM.jpg" $params -o $outputFileName  

				END=$(date +%s)
				DIFF=$(( $END - $START ))
				echo "Method computed in $DIFF "

			fi


			if [[ $evaluate = 1 ]];then
	
				# Evaluate Hausdorff
				#python $sourceDir/crownSegmenterEvaluator.py 0 $dataPrefix"/Z"$i"treetop.jpg" $outputFileName >> $hausFile

				# Evaluate matched percent
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataPrefix"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFile
				#python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataPrefix"/Z"$i"treetop.jpg" $roiMaskFileName $epsilon >> $reversePercentFile

				# Evaluate number of points difference
				python $sourceDir/crownSegmenterEvaluator.py 2 $dataPrefix"/Z"$i"treetop.jpg" $outputFileName $epsilon >> $pointDiffFile

				# Evaluate Euclidean
				#python $sourceDir/crownSegmenterEvaluator.py 4 $dataPrefix"/Z"$i"treetop.jpg" $outputFileName >> $eucFile


			fi
		done

		#echo "" >> $hausFile
		#echo "" >> $eucFile
		echo "" >> $percentFile
		#echo "" >> $reversePercentFile
		echo "" >> $pointDiffFile
	done

