#!/bin/bash

export RESDIR=./
export MOSESDIR=$HOME/opt/mosesdecoder
export BPEDIR=$HOME/opt/subword-nmt
export SLTKITDIR=./
export PYTHON3=python3

hostname

export modelName=asr.uni0
export systemName=how2
export sl=en
export tl=pt

export BASEDIR=$RESDIR/$systemName/
export BPESIZE=10000

echo $BASEDIR

##############   MT   #############################
mkdir -p $BASEDIR/data/orig/

cd $BASEDIR/data/orig/
mkdir -p parallel
mkdir -p valid
# training set
cd parallel
ln -s ../how2-300h-v1/data/train/text.pt how2.t
ln -s ../how2-300h-v1/data/train/text.en how2.s
# dev set
cd ../valid
ln -s ../how2-300h-v1/data/val/text.pt how2-val.t
ln -s ../how2-300h-v1/data/val/text.en how2-val.s
cd ..
# test set
mkdir eval/dev5 -p
cd eval/dev5
ln -s ../../how2-300h-v1/data/dev5/text.pt dev5.pt
ln -s ../../how2-300h-v1/data/dev5/text.en dev5.en
echo '*** Preprocessing original dataset'

$SLTKITDIR/scripts/defaultPreprocessor/Train.sh orig prepro

##############   ASR  #############################
cd $BASEDIR/data/prepro/train
ln -s ../../orig/how2-300h-v1/data/train/feats.scp how2.scp
cd -
cd $BASEDIR/data/prepro/valid
ln -s ../../orig/how2-300h-v1/data/val/feats.scp how2-val.scp
cd -
cd $BASEDIR/data/prepro/eval
ln -s ../../orig/how2-300h-v1/data/dev5/feats.scp dev5.scp

echo '*** Done'
