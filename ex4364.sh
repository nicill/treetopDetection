# /usr/bin/bash


sourceDir=$1
dataDir=$2
whatToDo=$3

outIMDir=$dataDir/outIm/
treeTopPredDir=$dataDir/bestDetections/
outputDir=$sourceDir/exper4/

mosaicString="1 2 3 4 5 6 7 8 9"
env1="crownSeg"
env2="fastai"

lRString="0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009"
#lRString="0.01 0.02"
#archString="res dense vgg squ alex"
#archString="dense"
classString="deciduous sickfir healthyfir"

epochs=10
augment=$5
frozen=$6
methodInput=$4"augment"$augment"Frozen"$frozen

archString=$7

trainAllSites=0
trainIndividualSites=1

echo $methodInput

summaryFile=$outputDir/$methodInput"summary.txt"

echo " make sure to prepare paramfile with correct paths!!!!"
#echo "data dir $dataDir"

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
	conda activate $env1
	bash reDoClassMasks.sh $dataDir
	conda deactivate

	for i in $mosaicString 
	do	
		mkdir $dataDir/patches$i
	done

	echo "Data preparation re-patching"
	conda activate $env2
	python $sourceDir/treeTopPatcher.py ./paramFile$augment"364" $augment
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
	cp ../patches9/*.* ./
	cd ..

	mkdir site4
	cd site4 
	cp ../patches1/*.* ./
	cp ../patches2/*.* ./
	cp ../patches3/*.* ./
	cp ../patches4/*.* ./
	cp ../patches5/*.* ./
	cp ../patches6/*.* ./
	cp ../patches7/*.* ./
	cp ../patches8/*.* ./
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
		conda activate $env2

		if [[ $train = 1 ]];then

			if [[ $trainAllSites = 1 ]];then
	
				echo "training!"
				START=$(date +%s)
				echo "models for "model"$net"LR"$lr"epochs"$epochs"frozen"$frozen "
				# TRAIN MODEL FOR ALL POSSIBLE SITE COMBINATIONS
				echo "training all sites"
				python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/AllSites/ "ASmodel"$net"LR"$lr"epochs"$epochs"frozen"$frozen $net t $epochs $lr $frozen
				END=$(date +%s)
				DIFF=$(( $END - $START ))
				echo "training all sites done in $DIFF "


			fi

			if [[ $trainIndividualSites = 1 ]];then

				echo "models for "model"$net"LR"$lr"epochs"$epochs"frozen"$frozen "

				siteString="1 2 3 4"
				for sit in $siteString
				do
					echo "trainig site $sit"
					START=$(date +%s)	
					python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site$sit/ "S"$sit"model"$net"LR"$lr"epochs"$epochs"frozen"$frozen $net t $epochs $lr $frozen
					END=$(date +%s)
					DIFF=$(( $END - $START ))
					echo "training site $sit done in $DIFF "

				done
			fi
		fi
		conda deactivate
	

		if [[ $compute = 1 ]];then

			echo "Testing (computing treetop prediction maps)"

			conda activate $env2

			#rm $sourceDir/tempImage*.*
	
			#first, do site 1
			siteString="1 2 3"
			for mos in $siteString
			do
				echo "python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site1/ "S1model"$net"LR"$lr"epochs"$epochs"frozen"$frozen $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/"
				python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site1/ "S1model"$net"LR"$lr"epochs"$epochs"frozen"$frozen $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/

			done
			#second, do site 2
			siteString="4 5 7"
			for mos in $siteString
			do
				echo "python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site2/ "S2model"$net"LR"$lr"epochs"$epochs"frozen"$frozen $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/"
				python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site2/ "S2model"$net"LR"$lr"epochs"$epochs"frozen"$frozen $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/
			done
	
			#third, do site 3
			siteString="6 8"
			for mos in $siteString
			do
				echo "python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site3/ "S3model"$net"LR"$lr"epochs"$epochs"frozen"$frozen $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/"
				python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site3/ "S3model"$net"LR"$lr"epochs"$epochs"frozen"$frozen $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/
			done

			#fourth, do site 4
			siteString="9"
			for mos in $siteString
			do
				echo "python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site4/ "S4model"$net"LR"$lr"epochs"$epochs"frozen"$frozen $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/"
				python $sourceDir/forestSpeciesPatchClassifierGenerateTreetopMasks.py $dataDir/site4/ "S4model"$net"LR"$lr"epochs"$epochs"frozen"$frozen $net p $treeTopPredDir/Z$mos"Best.jpg" $dataDir/Z$mos".jpg" $outIMDir/mosaic$mos/
			done

			conda deactivate
		fi


		if [[ $evaluate = 1 ]];then
			conda activate $env1

			echo -n "$net"LR"$lr"epochs"$epochs" >> $summaryFile

			siteString="1 2 3"
			for i in $siteString
			do
				roiMaskFileName=$dataDir"/Z"$i"ROI.jpg"
				epsilon=125

				outputFilePrefix=$outIMDir"/mosaic$i/S1model"$net"LR"$lr"epochs"$epochs"frozen"$frozen
				python $sourceDir/zaoExp4Processer.py $dataDir"/Z"$i $outputFilePrefix $epsilon $classString >> $summaryFile

				# Evaluate matched percent
				#class=deciduous
				#outputFileName=$outIMDir"/mosaic$i/S1model"$net"LR"$lr"epochs"$epochs"frozen"$frozen$class"tops.jpg"
				#python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i$class"tops.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFiledeciduous
				#python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i$class"tops.jpg" $roiMaskFileName $epsilon >> $reversePercentFiledeciduous

				#class=sickfir
				#outputFileName=$outIMDir"/mosaic$i/S1model"$net"LR"$lr"epochs"$epochs"frozen"$frozen$class"tops.jpg"
				#python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i$class"tops.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFilesickfir
				#python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i$class"tops.jpg" $roiMaskFileName $epsilon >> $reversePercentFilesickfir

				#class=healthyfir
				#outputFileName=$outIMDir"/mosaic$i/S1model"$net"LR"$lr"epochs"$epochs"frozen"$frozen$class"tops.jpg"
				#python $sourceDir/crownSegmenterEvaluator.py 1 $dataDir"/Z"$i$class"tops.jpg" $outputFileName $roiMaskFileName $epsilon >> $percentFilehealthyfir
				#python $sourceDir/crownSegmenterEvaluator.py 1 $outputFileName $dataDir"/Z"$i$class"tops.jpg" $roiMaskFileName $epsilon >> $reversePercentFilehealthyfir

			done

			siteString="4 5 7"
			for i in $siteString
			do
				roiMaskFileName=$dataDir"/Z"$i"ROI.jpg"
				epsilon=125
				outputFilePrefix=$outIMDir"/mosaic$i/S2model"$net"LR"$lr"epochs"$epochs"frozen"$frozen
				python $sourceDir/zaoExp4Processer.py $dataDir"/Z"$i $outputFilePrefix $epsilon $classString >> $summaryFile

			done

			siteString="6 8"
			for i in $siteString
			do
				roiMaskFileName=$dataDir"/Z"$i"ROI.jpg"
				epsilon=100
				outputFilePrefix=$outIMDir"/mosaic$i/S3model"$net"LR"$lr"epochs"$epochs"frozen"$frozen
				python $sourceDir/zaoExp4Processer.py $dataDir"/Z"$i $outputFilePrefix $epsilon $classString >> $summaryFile

			done


			siteString="9"
			for i in $siteString
			do
				roiMaskFileName=$dataDir"/Z"$i"ROI.jpg"
				epsilon=100

				outputFilePrefix=$outIMDir"/mosaic$i/S4model"$net"LR"$lr"epochs"$epochs"frozen"$frozen
				python $sourceDir/zaoExp4Processer.py $dataDir"/Z"$i $outputFilePrefix $epsilon $classString >> $summaryFile
			

			done

			echo "" >> $summaryFile
			conda deactivate

		fi
	done
done





if [[ $shutdown = 1 ]];then
shutdown
fi

