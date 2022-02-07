# /usr/bin/bash


sourceDir=$1
dataDir=$2
whatToDo=$3

outIMDir=$dataDir/outIm/
treeTopPredDir=$dataDir/bestDetections/
outputDir=$sourceDir/exp4/

mosaicString="1 2 3 4 5 6 7 8 9"
env1="crownSeg"
env2="pyt2"

#lRString="0.01 0.02"
lRString="0.01 0.02"
#archString="res vgg squ dense alex"
archString="res"
classString="deciduous sickfir healthyfir"

epochs=1
augment=10
methodInput=$4"augment"$augment


percentFiledeciduous=$outputDir/$methodInput"deciduouspercent.txt"
reversePercentFiledeciduous=$outputDir/$methodInput"deciduousreversePercent.txt"
percentFilesickfir=$outputDir/$methodInput"sickfirpercent.txt"
reversePercentFilesickfir=$outputDir/$methodInput"sickfirreversePercent.txt"
percentFilehealthyfir=$outputDir/$methodInput"healthyfirpercent.txt"
reversePercentFilehealthyfir=$outputDir/$methodInput"healthyfirreversePercent.txt"


echo "mosaic $mosaicString " > $percentFiledeciduous
echo "mosaic $mosaicString " > $reversePercentFiledeciduous
echo "mosaic $mosaicString " > $percentFilesickfir
echo "mosaic $mosaicString " > $reversePercentFilesickfir
echo "mosaic $mosaicString " > $percentFilehealthyfir
echo "mosaic $mosaicString " > $reversePercentFilehealthyfir

echo " make sure to prepare paramfile with correct paths!!!!"



# Codify the actions that the script will perform
dataPreparation=0
train=0
predict=0
evaluate=0
compute=0
shutdown=0
if [[ $whatToDo == *"d"* ]]; then
dataPreparation=1
fi
if [[ $whatToDo == *"t"* ]]; then
train=1
fi
if [[ $whatToDo == *"p"* ]]; then
predict=1
fi
if [[ $whatToDo == *"c"* ]]; then
compute=1
fi
if [[ $whatToDo == *"e"* ]]; then
evaluate=1
fi
if [[ $whatToDo == *"s"* ]]; then
shutdown=1
fi


if [[ $dataPreparation = 1 ]];then

	echo "Data preparation re-do Masks"
	#conda activate $env1
	#bash reDoClassMasks.sh $dataDir
	#conda deactivate

	for i in $mosaicString 
	do	
		mkdir $dataDir/patches$i
	done

	echo "Data preparation re-patching"
	conda activate $env2
	python $sourceDir/treeTopPatcher.py ./paramFile $augment
	conda deactivate

	echo "Data preparation re-making site folders"
	# create one folder per site, also create one folder for all images
	cd $dataDir

	mkdir site1
	cd site1 
	cp ../patches4/*.* ./
	cp ../patches5/*.* ./
	cp ../patches6/*.* ./
	cp ../patches7/*.* ./
	cp ../patches8/*.* ./
	cp ../patches9/*.* ./
	cd ..

	mkdir site2
	cd site2 
	cp ../patches1/*.* ./
	cp ../patches2/*.* ./
	cp ../patches3/*.* ./
	cp ../patches6/*.* ./
	cp ../patches8/*.* ./
	cp ../patches9/*.* ./
	cd ..

	mkdir site3
	cd site3 
	cp ../patches1/*.* ./
	cp ../patches2/*.* ./
	cp ../patches3/*.* ./
	cp ../patches4/*.* ./
	cp ../patches5/*.* ./
	cp ../patches7/*.* ./
	cd ..

	mkdir AllSites
	cd AllSites 
	cp ../patches1/*.* ./
	cp ../patches2/*.* ./
	cp ../patches3/*.* ./
	cp ../patches4/*.* ./
	cp ../patches5/*.* ./
	cp ../patches6/*.* ./
	cp ../patches7/*.* ./
	cp ../patches8/*.* ./
	cp ../patches9/*.* ./
	cd ..

	cd $sourceDir
	pwd
fi


if [[ $compute = 1 ]];then

	mkdir $outIMDir/

	for i in $mosaicString 
	do	
		mkdir $outIMDir/mosaic$i
	done	
fi

if [[ $evaluate = 1 ]];then

	mkdir $outputDir/
fi

for net in $archString 
do
	for lr in $lRString
	do

		if [[ $train = 1 ]];then
			conda activate $env2
	
			echo "training!"

			echo "models for "model"$net"LR"$lr"epochs"$epochs "
			# TRAIN MODEL FOR ALL POSSIBLE SITE COMBINATIONS
			#python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/AllSites/ "ASmodel"$net"LR"$lr"epochs"$epochs $net t $epochs $lr
			siteString="1 2 3"
			for sit in $siteString
			do
				python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site$sit/ "S"$sit"model"$net"LR"$lr"epochs"$epochs $net t $epochs $lr
			done
			conda deactivate
		fi

		if [[ $compute = 1 ]];then
			conda activate $env2

			echo " "
			
			#first, do site 1
			siteString="1 2 3"
			for mos in $siteString
			do
				echo "python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site1/ "S1model"$net"LR"$lr"epochs"$epochs $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/"
				python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site1/ "S1model"$net"LR"$lr"epochs"$epochs $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/

			done
			#second, do site 2
			siteString="3 4 7"
			for mos in $siteString
			do
				echo "python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site2/ "S2model"$net"LR"$lr"epochs"$epochs $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/"
				python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site2/ "S2model"$net"LR"$lr"epochs"$epochs $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/
			done
	
			#third, do site 3
			siteString="6 8 9"
			for mos in $siteString
			do
				echo "python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site3/ "S3model"$net"LR"$lr"epochs"$epochs $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/"
				python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site3/ "S3model"$net"LR"$lr"epochs"$epochs $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/
			done

			conda deactivate
		fi


		if [[ $evaluate = 1 ]];then
			conda activate $env1

			siteString="1 2 3"
			for i in $siteString
			do
				roiMaskFileName=$dataDir"/Z"$i"ROI.jpg"
				epsilon=125

				# Evaluate matched percent
				class=deciduous
				outputFileName=$outIMDir"/mosaic$i/S1model"$net"LR"$lr"epochs"$epochs$class"tops.jpg"
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFiledeciduous
				python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i"treetop.jpg" $roiMaskFileName $epsilon >> $reversePercentFiledeciduous

				class=sickfir
				outputFileName=$outIMDir"/mosaic$i/S1model"$net"LR"$lr"epochs"$epochs$class"tops.jpg"
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFilesickfir
				python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i"treetop.jpg" $roiMaskFileName $epsilon >> $reversePercentFilesickfir

				class=healthyfir
				outputFileName=$outIMDir"/mosaic$i/S1model"$net"LR"$lr"epochs"$epochs$class"tops.jpg"
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFilehealthyfir
				python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i"treetop.jpg" $roiMaskFileName $epsilon >> $reversePercentFilehealthyfir

			done

			siteString="4 5 7"
			for i in $siteString
			do
				roiMaskFileName=$dataDir"/Z"$i"ROI.jpg"
				epsilon=125

				# Evaluate matched percent
				class=deciduous
				outputFileName=$outIMDir"/mosaic$i/S2model"$net"LR"$lr"epochs"$epochs$class"tops.jpg"
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFiledeciduous
				python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i"treetop.jpg" $roiMaskFileName $epsilon >> $reversePercentFiledeciduous

				class=sickfir
				outputFileName=$outIMDir"/mosaic$i/S2model"$net"LR"$lr"epochs"$epochs$class"tops.jpg"
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFilesickfir
				python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i"treetop.jpg" $roiMaskFileName $epsilon >> $reversePercentFilesickfir

				class=healthyfir
				outputFileName=$outIMDir"/mosaic$i/S2model"$net"LR"$lr"epochs"$epochs$class"tops.jpg"
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFilehealthyfir
				python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i"treetop.jpg" $roiMaskFileName $epsilon >> $reversePercentFilehealthyfir

			done


			siteString="6 8 9"
			for i in $siteString
			do
				roiMaskFileName=$dataDir"/Z"$i"ROI.jpg"
				epsilon=100

				# Evaluate matched percent
				class=deciduous
				outputFileName=$outIMDir"/mosaic$i/S3model"$net"LR"$lr"epochs"$epochs$class"tops.jpg"
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFiledeciduous
				python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i"treetop.jpg" $roiMaskFileName $epsilon >> $reversePercentFiledeciduous

				class=sickfir
				outputFileName=$outIMDir"/mosaic$i/S3model"$net"LR"$lr"epochs"$epochs$class"tops.jpg"
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFilesickfir
				python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i"treetop.jpg" $roiMaskFileName $epsilon >> $reversePercentFilesickfir

				class=healthyfir
				outputFileName=$outIMDir"/mosaic$i/S3model"$net"LR"$lr"epochs"$epochs$class"tops.jpg"
				python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i"treetop.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFilehealthyfir
				python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i"treetop.jpg" $roiMaskFileName $epsilon >> $reversePercentFilehealthyfir


			done

			echo "" >> $percentFiledeciduous
			echo "" >> $reversePercentFiledeciduous
			echo "" >> $percentFilesickfir
			echo "" >> $reversePercentFilesickfir
			echo "" >> $percentFilehealthyfir
			echo "" >> $reversePercentFilehealthyfir
			conda deactivate

		fi
	done
done





if [[ $shutdown = 1 ]];then
shutdown
fi

