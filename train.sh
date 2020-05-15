#!/bin/bash
export RESDIR=./
export NMTDIR=./
export SLTKITDIR=./
export PYTHON3=python3
export GPU=0 # if CPU: -1

export modelName=asr.uni0
export systemName=how2
export sl=en
export tl=pt

export BASEDIR=$RESDIR/how2/
export BPESIZE=10000

export LAYER=12
export TRANSFORMER=stochastic_transformer #stochastic

echo $BASEDIR
export ENC_LAYER=32

# ASR
$SLTKITDIR/scripts/NMTGMinor/Train.speech.uni0.sh prepro $modelName s 
# Speech translation
$SLTKITDIR/scripts/NMTGMinor/Train.speech.uni0.sh prepro $modelName t
