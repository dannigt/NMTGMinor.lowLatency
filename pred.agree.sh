#!/bin/bash
export RESDIR=./
export NMTDIR=./
export SLTKITDIR=./
export GPU=0

export sl=en
export tl=pt

export modelName=asr.uni0

export BASEDIR=$RESDIR/$systemName/


echo $BASEDIR


for tst in dev5.0.5sec; do
	for strategy in agree2; do
		export BEAMSIZE=8
	        $SLTKITDIR/scripts/NMTGMinor/Translate.speech.agree.sh $tst prepro $modelName $strategy

		for beam in .beam8. ; do
        		echo $BASEDIR/data/$modelName/eval/${tst}${beam}$strategy.t

			# Postprocess  
			sed -e "s/@@ //g"  $BASEDIR/data/$modelName/eval/${tst}${beam}$strategy.t | sed -e "s/@@$//g" | sed -e "s/&apos;/'/g" -e 's/&#124;/|/g' -e "s/&amp;/&/g" -e 's/&lt;/>/g' -e 's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#91;/[/g' -e 's/&#93;/]/g' | sed -e "s/ '/'/g" | sed -e "s/\.//" -e "s/,//g" -e "s/\!//g" -e "s/?//g" -e "s/;//g" -e "s/://g" -e "s/\.//g" -e "s/--/ /g"  -e "s/-/ /" |  perl -nle 'print lc' >  $BASEDIR/data/$modelName/eval/${tst}${beam}$strategy.asr

			# Copy utterance ID to transform to .sclite
			awk '{print "(" $1 ")"}'  $BASEDIR/data/orig/how2-300h-v1/data/dev5/text.id.en | paste $BASEDIR/data/$modelName/eval/${tst}${beam}$strategy.asr - > $BASEDIR/data/$modelName/eval/${tst}${beam}$strategy.sclite

			# Filter reference file
			cat $BASEDIR/data/orig/how2-300h-v1/data/dev5/text.en | sed -e "s/&apos;/'/g" -e 's/&#124;/|/g' -e "s/&amp;/&/g" -e 's/&lt;/>/g' -e 's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#91;/[/g' -e 's/&#93;/]/g' | sed -e "s/ '/'/g" | sed -e "s/\.//" -e "s/,//g" -e "s/\!//g" -e "s/?//g" -e "s/;//g" -e "s/://g" -e "s/\.//g" -e "s/--/ /g"  -e "s/-/ /" |  perl -nle 'print lc' > $BASEDIR/data/$modelName/eval/text.en.filtered
		
			# Copy utterance ID to transform to .sclite
			awk '{print "(" $1 ")"}'  $BASEDIR/data/orig/how2-300h-v1/data/dev5/text.id.en | paste $BASEDIR/data/$modelName/eval/text.en.filtered - > $BASEDIR/data/$modelName/eval/text.en.sclite

			# Eval ASR
			echo "*** WER"
			~/opt/sctk-2.4.10/bin/sclite  -r  $BASEDIR/data/$modelName/eval/text.en.sclite -h $BASEDIR/data/$modelName/eval/${tst}${beam}$strategy.sclite -i spu_id -f 0 -o sum stdout dtl pra | grep Sum/Avg | awk '{print $11}'

			python3 ~/src/NMTGMinor.private/smalltools/analyze_latency.py $BASEDIR/data/$asr/eval/${tst}${beam}$strategy.t.latency

			# Eval SLT
			export modelName=asr.uni0.slt
			echo "*** BLEU"
			cat $BASEDIR/data/$modelName/eval/dev5${beam}$strategy.pt | sacrebleu $BASEDIR/data/orig/how2-300h-v1/data/dev5/text.$strategy.pt

			echo "*** METEOR"
	                nmtpy-coco-metrics $BASEDIR/data/$modelName/eval/dev5${beam}$strategy.pt -r $BASEDIR/data/orig/how2-300h-v1/data/dev5/text.pt
		done;
        done;
done
