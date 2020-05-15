#!/bin/bash

set=$1
input=$2
name=$3
strategy=$4
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
    out=$BASEDIR/data/$name/eval/$set.$strategy.t
else
    out=$BASEDIR/data/$name/eval/$set.beam$BEAMSIZE.$strategy.t
fi

echo 'decoding with beam size ' $BEAMSIZE
echo $BASEDIR/model/$name/$model

python3 -u $NMTDIR/translate.py $gpu_string \
       -model $BASEDIR/model/$name/$model \
       -src $BASEDIR/data/$input/eval/$set.scp \
       -batch_size 1 -verbose\
       -beam_size $BEAMSIZE -alpha 1.0 \
       -encoder_type audio -asr_format scp -concat 4 \
       -normalize \
       -output $out \
       -output_latency $out.latency \
       -output_confidence $out.conf \
       -start_prefixing_from 2 \
       -require_prefix_agreements 2
