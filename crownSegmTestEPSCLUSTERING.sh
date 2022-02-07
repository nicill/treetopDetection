# /usr/bin/bash


#this needs to be called from inside a proper python environment!

#Example call
#./testScripts/crownSegmTest.sh /media/yago/workDrive/Experiments/forests/segmentation /media/yago/workDrive/Experiments/forests/segmentation/testScripts/ /media/yago/workDrive/Experiments/forests/segmentation/crown ce 012345 100

sourceDir=$1
outputDir=$2
dataPrefix=$3
whatToDo=$4
methodInput=$5
epsilon=$6

mosaicString="1 2 3 4 5 6 7 8 9"

if [[ $methodInput == *"0"* ]]; then
methodString="$methodString 0"
fi
if [[ $methodInput == *"1"* ]]; then
methodString="$methodString 1"
fi
if [[ $methodInput == *"2"* ]]; then
methodString="$methodString 2"
fi
if [[ $methodInput == *"3"* ]]; then
methodString="$methodString 3"
fi
if [[ $methodInput == *"4"* ]]; then
methodString="$methodString 4"
fi
if [[ $methodInput == *"5"* ]]; then
methodString="$methodString 5"
fi


refine="yes"

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

echo "method $mosaicString " > $hausFile
echo "method $mosaicString " > $eucFile
echo "method $mosaicString " > $percentFile
echo "method $mosaicString " > $reversePercentFile
echo "method $mosaicString " > $pointDiffFile

# Main loop
for method in $methodString
do

	# Change parameters for the different methods

	if [[ $method = 0 ]];then

		wSizes="150 200 250 275 300 350 400 450 500 750 1000"
		declare -a paramVector=()	
		declare -a paramStringVector=()
		for ws in $wSizes
		do
			paramStringVector+=("s$ws")
			paramVector+=("-s $ws ")
		done
	elif [[ $method = 1 ]] ;then
		declare -a paramVector=()	
		declare -a paramStringVector=()	

		wSizes="250 400 500 750 1000 1500"
		thresholds="75 100 150"
		minDists="50 75 100 120"
		for ws in $wSizes
		do
			for th in $thresholds
			do
				for md in $minDists
				do

					paramStringVector+=("s$ws""th$th""md$md")
					paramVector+=("-s $ws -th $th -md $md")

				done
			done
		done
		#echo ${paramVector[@]}
	elif [[ $method = 2 ]];then

		# window size affects, epsilon too 
		wSizes="150 200 250 275 300 350 400 450"
		epsilons="0.5 1 2 3 4 5 10"
		minS="5 10 15 20 25 30" 	
		declare -a paramVector=()	
		declare -a paramStringVector=()
		for ws in $wSizes
		do
			for eps in $epsilons
			do

				for ms in $minS
				do
					paramStringVector+=("s$ws""eps$eps""mins$ms")
					paramVector+=("-s $ws -eps $eps -ms $ms")

				done
			done
		done	
	elif [[ $method = 3 ]];then
		declare -a paramVector=()	
		declare -a paramStringVector=()	

		# Threshold affects, window size too, the size in the maximum filter does not seem to affect

		wSizes="400 500 750 200 250 300"
		#nClust="1 2 3 5 7 10 15 20"
		nClust="5"
		for ws in $wSizes
		do
			for nc in $nClust
			do
				paramStringVector+=("s$ws""nc$nc")
				paramVector+=("-s $ws -nc $nc")
			done
		done
	elif [[ $method = 4 ]];then
		declare -a paramVector=()	
		declare -a paramStringVector=()	

		# Threshold affects, window size too, the size in the maximum filter does not seem to affect

		wSizes="200 250 300 400 500 750"
		nClust="5"
		for ws in $wSizes
		do
			for nc in $nClust
			do
				paramStringVector+=("s$ws""nc$nc")
				paramVector+=("-s $ws -nc $nc")
			done
		done
	elif [[ $method = 5 ]];then
		declare -a paramVector=()	
		declare -a paramStringVector=()	

		# Threshold affects, window size too, the size in the maximum filter does not seem to affect

		#wSizes="100 150 200 250 300 250 400 500 750 1000 2000 5000"
		wSizes="200 250 300 400 500 750"
		nClust="5"
		for ws in $wSizes
		do
			for nc in $nClust
			do
				paramStringVector+=("s$ws""nc$nc")
				paramVector+=("-s $ws -nc $nc")
			done
		done
		#echo ${paramStringVector[@]}

	fi

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
			outputFileName=$dataPrefix"/DEM"$i"_binaryMethod"$method$paramString"Radius"$refineRadius"Percentile"$percentile".jpg"
			roiMaskFileName=$dataPrefix"/Z"$i"ROI.jpg"

			#echo "outputFile $outputFileName " 
			
			if [[ $compute = 1 ]];then
		
				START=$(date +%s)			

				python $sourceDir/sliding_windowCLUSTERING.py -m $method -i $dataPrefix"/Z"$i"nDEM.jpg" -d $dataPrefix"/Z"$i"nDEM.jpg" -w y $params -o $outputFileName -grad yes -ref $refine -pf 50

				# Used to be
				#python $sourceDir/sliding_window.py -m $method -i $dataPrefix"/mosaic"$i".jpg" -d $dataPrefix"/DEM"$i".jpg" -w y $params -f $floor -o $outputFileName -ref $refine -lf $dataPrefix"/mosaic"$i"labels.jpg" -pf 50

				END=$(date +%s)
				DIFF=$(( $END - $START ))
				echo "Method computed in $DIFF "

			fi


			if [[ $evaluate = 1 ]];then
	
	
				# Evaluate Hausdorff
				python $sourceDir/crownSegmenterEvaluator.py 0 $dataPrefix"/Z"$i"nDEM.jpg" $outputFileName >> $hausFile

				# Evaluate matched percent
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataPrefix"/Z"$i"nDEM.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFile
				python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataPrefix"/Z"$i"nDEM.jpg" $roiMaskFileName $epsilon >> $reversePercentFile

				# Evaluate number of points difference
				python $sourceDir/crownSegmenterEvaluator.py 2 $dataPrefix"/Z"$i"nDEM.jpg" $outputFileName $epsilon >> $pointDiffFile

				# Evaluate Euclidean
				python $sourceDir/crownSegmenterEvaluator.py 4 $dataPrefix"/Z"$i"nDEM.jpg" $outputFileName >> $eucFile


			fi
		done

		echo "" >> $hausFile
		echo "" >> $eucFile
		echo "" >> $percentFile
		echo "" >> $reversePercentFile
		echo "" >> $pointDiffFile
	done
done
