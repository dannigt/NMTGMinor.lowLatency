# NMTGMinor.lowLatency


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
`./pred.sh`

### Chunk-based incremental decoding


