#!/bin/bash

set=$1
input=$2
name=$3
prefix_from=$4
remove_last_n=$5

if [ "$#" -ne 5 ]; then
    remove_last_n=0
fi

model=model.pt

if [ -z "$BASEDIR" ]; then
    BASEDIR=/
fi

if [ -z "$NMTDIR" ]; then
    NMTDIR=/opt/NMTGMinor/
fi

if [ -z "$GPU" ]; then
    GPU=0
fi

if [ -z "$BEAMSIZE" ]; then
    BEAMSIZE=1
fi


if [ $GPU == -1 ]; then
    gpu_string=""
else
    gpu_string="-gpu "$GPU
fi

mkdir -p $BASEDIR/data/$name/eval/

if [ $BEAMSIZE == 1 ]; then
    out=$BASEDIR/data/$name/eval/$set.prefix.from.$prefix_from.remove.$remove_last_n.t
else
    out=$BASEDIR/data/$name/eval/$set.beam$BEAMSIZE.prefix.from.$prefix_from.remove.$remove_last_n.t
fi

echo 'decoding with beam size ' $BEAMSIZE
echo $BASEDIR/model/$name/$model

python3 -u $NMTDIR/translate.py $gpu_string \
       -model $BASEDIR/model/$name/$model \
       -src $BASEDIR/data/$input/eval/$set.scp \
       -batch_size 1  -verbose\
       -beam_size $BEAMSIZE -alpha 1.0 \
       -encoder_type audio -asr_format scp -concat 4 \
       -normalize \
       -output $out \
       -start_prefixing_from $prefix_from \
       -output_latency $out.latency \
       -output_confidence $out.conf \
       -remove_last_n $remove_last_n

