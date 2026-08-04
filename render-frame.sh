#!/bin/bash

if [ "$1" == "" ]; then
    echo "Usage: $(basename $0) ANGLE"
    exit 1
fi

readonly angle="$1"
readonly angleNNN=$(printf "%03d" $angle)
readonly pfmfile=image$angleNNN.pfm
readonly pngfile=image$angleNNN.png
time python3 -m pytracer demo --algorithm pathtracing --samples-per-pixel 9 --num-of-rays 10 --max-depth 2 --angle-deg $angle --width 640 --height 360 --pfm-output $pfmfile && python3 -m pytracer pfm2png --luminosity 0.5 $pfmfile $pngfile
