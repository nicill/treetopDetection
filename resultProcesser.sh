
firstBandPercentString="75 80 85 90"

yesnoString="yes"
refRadString="40 50 75"

for band in $firstBandPercentString;
do
	for global in $yesnoString;
	do
		for refRad in $refRadString;
		do
			echo "$global$refRad"FBP"$band"percent.txt""
			python processOutput.py ./EXP1/topsRefine$global$refRad"FBP"$band"percent.txt" ./EXP1/topsRefine$global$refRad"FBP"$band"pointDiff.txt"
		done
	done
done




firstBandPercentString="20 50 60"


yesnoString="yes"
refRadString="40 50"

for band in $firstBandPercentString;
do


	for global in $yesnoString;
	do
		for refRad in $refRadString;
		do
			echo "PERMISSIVE$global$refRad"FBP"$band"percent.txt""
			python processOutput.py ./EXP1/PermissivetopsRefine$global$refRad"FBP"$band"percent.txt" ./EXP1/PermissivetopsRefine$global$refRad"FBP"$band"pointDiff.txt"

		done
	done

done
