# /usr/bin/bash


#this needs to be called from inside a proper python environment!

#Example call
#./testScripts/crownSegmTest.sh /media/yago/workDrive/Experiments/forests/segmentation /media/yago/workDrive/Experiments/forests/segmentation/testScripts/ /media/yago/workDrive/Experiments/forests/segmentation/crown ce 

sourceDir=$1
outputDir=$2
dataPrefix=$3
whatToDo=$4
methodInput=$5
refine=$6
refineRadius=$7
percentile=$8


mosaicString="1 2 3 4 5 6 7 8 9"


hausFile=$outputDir/$methodInput"haus.txt"
eucFile=$outputDir/$methodInput"euc.txt"
percentFile=$outputDir/$methodInput"percent.txt"
reversePercentFile=$outputDir/$methodInput"reversePercent.txt"
pointDiffFile=$outputDir/$methodInput"pointDiff.txt"
eucMatchedFile=$outputDir/$methodInput"eucMatched.txt"
repeatedFile=$outputDir/$methodInput"repeated.txt"

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

echo "method $mosaicString " > $hausFile
echo "method $mosaicString " > $eucFile
echo "method $mosaicString " > $percentFile
echo "method $mosaicString " > $reversePercentFile
echo "method $mosaicString " > $pointDiffFile
echo "method $mosaicString " > $eucMatchedFile
echo "method $mosaicString " > $repeatedFile


	wSizes="500 750 1000"
	thresholds="500 750 1000 1500 2000 2500"
	minDists="0.1 0.25 0.4 0.5 0.75"
	for ws in $wSizes
	do
		for th in $thresholds
		do
			for md in $minDists
			do

				paramStringVector+=("s$ws""mpt$th""ts$md")
				paramVector+=("-s $ws -mpt $th -ts $md")

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
			outputFileName=$dataPrefix"/DSM"$i"_binaryMethod"$method$paramString"Radius"$refineRadius"Percentile"$percentile".jpg"
			roiMaskFileName=$dataPrefix"/Z"$i"ROI.jpg"

			#echo "outputFile $outputFileName " 
			
			if [[ $compute = 1 ]];then
		
				START=$(date +%s)
				echo "python $sourceDir/sliding_window.py -d $dataPrefix"/Z"$i"nDSM.tif" $params -o $outputFileName -refine $refine -refRad $refineRadius"			

				python $sourceDir/sliding_window.py -d $dataPrefix"/Z"$i"nDSM.tif" $params -o $outputFileName  -ref $refine -refRad $refineRadius -perc $percentile

				END=$(date +%s)
				DIFF=$(( $END - $START ))
				echo "Method computed in $DIFF "

			fi


			if [[ $evaluate = 1 ]];then
	
				if [ $i -eq 1 -o $i -eq 2 -o $i -eq 3 -o $i -eq 4 -o $i -eq 5 -o $i -eq 7 ]; then
					epsilon=125
				fi
				if [ $i -eq 6 -o $i -eq 8 -o $i -eq 9 ]; then
					epsilon=100
				fi
	
				# Evaluate Hausdorff
				python $sourceDir/crownSegmenterEvaluator.py 0 $dataPrefix"/Z"$i"treetop.jpg" $outputFileName >> $hausFile

				# Evaluate matched percent
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataPrefix"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFile
				python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataPrefix"/Z"$i"treetop.jpg" $roiMaskFileName $epsilon >> $reversePercentFile

				# Evaluate number of points difference
				python $sourceDir/crownSegmenterEvaluator.py 2 $dataPrefix"/Z"$i"treetop.jpg" $outputFileName $epsilon >> $pointDiffFile

				# Evaluate Euclidean
				python $sourceDir/crownSegmenterEvaluator.py 4 $dataPrefix"/Z"$i"treetop.jpg" $outputFileName >> $eucFile

				#euclidean matched
				python $sourceDir/crownSegmenterEvaluator.py 5 $dataPrefix"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $eucMatchedFile

				#repeated count
				python $sourceDir/crownSegmenterEvaluator.py 6 $dataPrefix"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $repeatedFile

			fi
		done

		echo "" >> $hausFile
		echo "" >> $eucFile
		echo "" >> $percentFile
		echo "" >> $reversePercentFile
		echo "" >> $pointDiffFile
		echo "" >> $eucMatchedFile
		echo "" >> $repeatedFile

	done
done
