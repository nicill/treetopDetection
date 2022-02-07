
firstBandPercent=$1

yesnoString="yes"
refRadString="40 50 75 100"
#refRadString="40"

for global in $yesnoString;
do
	for refRad in $refRadString;
	do
		echo " oi"
		gnome-terminal -e "bash crownSegmTestEPS.sh /home/owner/Experiments/forests/crownSegm2 /home/owner/Experiments/forests/crownSegm2/reOutput100 ./otherData e topsRefine$global$refRad"FBP"$firstBandPercent 100 $global $refRad $firstBandPercent"
	done
done

exit

yesnoString="yes"
refRadString="40 50 60"

for global in $yesnoString;
do
	for refRad in $refRadString;
	do
		echo " oi"
		gnome-terminal -e "bash crownSegmTestEPS2.sh /home/owner/Experiments/forests/crownSegm2 /home/owner/Experiments/forests/crownSegm2/reOutput100/ ./otherData e PermissivetopsRefine$global$refRad"FBP"$firstBandPercent 100 $global $refRad $firstBandPercent"
	done
done

exit







