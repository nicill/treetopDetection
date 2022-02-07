# /usr/bin/bash

datadir=$1

mosaicString="1 2 3 4 5 6 7 8 9"
classNames="sickfir healthyfir deciduous"

for class in $classNames
do
	for mosaic in $mosaicString
	do
		echo "doing python demUtils.py 5 $datadir/"Z"$mosaic"treetop.jpg" $datadir/"Z"$mosaic$class".jpg" $datadir/"Z"$mosaic$class"tops.jpg""S
		python demUtils.py 5 $datadir/"Z"$mosaic"treetop.jpg" $datadir/"Z"$mosaic$class".jpg" $datadir/"Z"$mosaic$class"tops.jpg"
	done
done	
