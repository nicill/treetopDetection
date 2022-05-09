# /usr/bin/bash


#this needs to be called from inside a proper python environment!

#Example call
#bash crownSegmTestEPSHerve.sh /media/yago/workDrive/Experiments/forests/treetopDetection/ /media/yago/workDrive/Experiments/forests/treetopDetection/output/ /media/yago/workDrive/Experiments/forests/treetopDetection/Herve/lowres/ ce ONEBAND yes 1

sourceDir=$1
outputDir=$2
dataPrefix=$3
whatToDo=$4
methodInput=$5
refine=$6
refineRadius=$7
#percentile=$8


#mosaicString="site1_mav_ft site1_mav_nft site1_p4_ft site1_p4_nft"
#mosaicString="site1_mav_ft site1_mav_nft"
#mosaicString="site2_p4_ft site2_p4_nft"
mosaicString="site2_p4_nft"

#epsilonString="3 6 8 10 12"	
epsilonString="3 5 7 10"	

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


	wSizes="75 100 250 500 1000"
	thresholds="1 2 5 10"
	minDists="0.1 0.5 1"
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
		echo -n "$method $params " >> $eucMatchedFile
		echo -n "$method $params " >> $percentFile
		echo -n "$method $params " >> $reversePercentFile
		echo -n "$method $params " >> $pointDiffFile
		echo -n "$method $params " >> $repeatedFile

		for i in $mosaicString
		do
				
			echo "doing method $method with mosaic $i and params $params "
			outputFileName=$dataPrefix"/OUT"$i"_binaryMethod"$method$paramString"Radius"$refineRadius"Percentile"$percentile".png"
			#roiMaskFileName=$dataPrefix"/"$i"_fir.png"
			roiMaskFileName=$dataPrefix"/"$i"_cedar.png"

			echo $roiMaskFileName 

			#echo "outputFile $outputFileName " 
			
			if [[ $compute = 1 ]];then
		
				START=$(date +%s)
				#python $sourceDir/sliding_window.py -d"$dataPrefix"/"$i"_chm.tif $params -o $outputFileName -perc $percentile -ref $refine -refRad $refineRadius

				echo "python $sourceDir/sliding_window.py -d" $dataPrefix"/"$i"_chm.tif $params -o $outputFileName -ref $refine -refRad $refineRadius"			
				python $sourceDir/sliding_window.py -d"$dataPrefix"/"$i"_chm.tif $params -o $outputFileName  -ref $refine -refRad $refineRadius

				END=$(date +%s)
				DIFF=$(( $END - $START ))
				echo "Method computed in $DIFF "

			fi


			if [[ $evaluate = 1 ]];then

				for epsilon in $epsilonString
				do
					# Evaluate matched percent
					python $sourceDir/crownSegmenterEvaluator.py 1 $dataPrefix"/"$i"_tops.png" $outputFileName $roiMaskFileName $epsilon >> $percentFile
					#python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataPrefix"/"$i"_tops.png" $roiMaskFileName $epsilon >> $reversePercentFile

					#euclidean matched
					#python $sourceDir/crownSegmenterEvaluator.py 5 $dataPrefix"/"$i"_tops.png" $outputFileName $roiMaskFileName $epsilon >> $eucMatchedFile

					#repeated count
					python $sourceDir/crownSegmenterEvaluator.py 6 $dataPrefix"/"$i"_tops.png" $outputFileName $roiMaskFileName $epsilon >> $repeatedFile
				done


				# Evaluate Hausdorff
				#python $sourceDir/crownSegmenterEvaluator.py 0 $dataPrefix"/"$i"_tops.png" $outputFileName >> $hausFile

				# Evaluate number of points difference
				python $sourceDir/crownSegmenterEvaluator.py 2 $dataPrefix"/"$i"_tops.png" $outputFileName $roiMaskFileName >> $pointDiffFile

				# Evaluate Euclidean
				#python $sourceDir/crownSegmenterEvaluator.py 4 $dataPrefix"/"$i"_tops.png" $outputFileName >> $eucFile


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

