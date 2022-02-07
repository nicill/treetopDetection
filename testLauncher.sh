
firstBandPercent=$1

yesnoString="yes"
refRadString="40 50 75 100"

for global in $yesnoString;
do
	for refRad in $refRadString;
	do
		echo " oi"
		gnome-terminal -e "bash crownSegmTestEPS.sh /home/owner/Experiments/forests/crownSegm/ /home/owner/Experiments/forests/crownSegm/output/ /home/owner/Experiments/forests/crownSegm/data/ ce topsRefine$global$refRad"FBP"$firstBandPercent $global $refRad $firstBandPercent"

	done
done


