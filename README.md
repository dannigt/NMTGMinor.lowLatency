# NMTGMinor.lowLatency
Based on https://github.com/quanpn90/NMTGMinor and https://github.com/jniehues-kit/SLT.KIT

## Requirements
* Python > 3.6
* pytorch
* hdf5
* apex
* kaldiio
* sacreBleu

## Data prep
Get pre-packaged data via https://github.com/srvk/how2-dataset

Place the dataset under `$BASEDIR/data/orig/how2-300h-v1`

Run preprocessor:

`./prep.data.sh`

## Train 
Train a unidirectional model by:

`./train.sh`

where `NMTGMinor/Train.speech.uni0.sh` gets called. `-limit_rhs_steps` controls the number of look-ahead steps. 
`-limit_rhs_steps=0` means no look-ahead.

## Predict & Eval
### Full-sequence decoding
Run prediction and eval by:

`./pred.sh`

### Chunk-based incremental decoding
First chunk the input utterances by:

`python ./smalltools/create_chunks.py $RESDIR/how2/data/prepro/eval/dev5.scp $RESDIR/how2/data/prepro/eval_partial_0.5sec 50`

Point to the partial utterances in test set:
`ln -s $RESDIR/how2/data/prepro/eval_partial_0.5sec/feats.scp $RESDIR/how2/data/prepro/eval/dev5.0.5sec.scp`
`ln -s $RESDIR/how2/data/prepro/eval_partial_0.5sec/num.partial.seqs.0.5sec.pickle $RESDIR/how2/data/prepro/eval/dev5.0.5sec.num.partial.seqs.pickle`

Run prediction and eval under strategy local agreement:

`./pred.agree.sh`

Run prediction and eval under strategy hold-n:

`./pred.holdn.sh`

Run prediction and eval under strategy wait-k:

`./pred.waitk.sh`


