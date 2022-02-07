arch=$1
GPU=$2
what=$3
prefix=$4

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@starting exp4 A with $arch unfrozen no augment"
START=$(date +%s)
date
	CUDA_VISIBLE_DEVICES=$GPU bash ./ex4364.sh /home/owner/Experiments/forests/crownSegm/ /home/owner/Experiments/forests/crownSegm/exper4Data/noaugment364/ $what exp4UNFNoaugm364$arch 0 False $arch >> outExp4$prefix$arch.txt 
date
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "calculations for $arch done in $DIFF "

augmentString="2 6 10 20"

for aug in $augmentString
do

	echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@starting exp4 A with $arch unfrozen augment $aug"
	START=$(date +%s)
	date
		CUDA_VISIBLE_DEVICES=$GPU bash ./ex4364.sh /home/owner/Experiments/forests/crownSegm/ /home/owner/Experiments/forests/crownSegm/exper4Data/augment$aug"364"/ $what exp4UNF$aug"364"$arch $aug False $arch >> outExp4$prefix$arch.txt 
	date
	END=$(date +%s)
	DIFF=$(( $END - $START ))
	echo "calculations for $arch done in $DIFF "



done
