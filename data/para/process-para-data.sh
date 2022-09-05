#!/bin/sh

set -u

SCRIPT=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
MAIN_DIR=$(realpath "$SCRIPT_DIR/../..")

N_WORKERS=16

# parallel data dir
PARA_DIR=$MAIN_DIR/data/para

# spm
SPM_MODEL=$MAIN_DIR/PTModels/Liv4ever-MT/enlvetli.model
SPM_ENCODE=$MAIN_DIR/fairseq/scripts/spm_encode.py
DICT=$MAIN_DIR/PTModels/M2M100-CMEA/merge_dict.txt

echo "processing authentic data"
lng_pairs=(en-et en-liv en-lv et-liv et-lv liv-lv)
for lng_pair in ${lng_pairs[*]}
do
    echo "processing $lng_pair..."

    src_lng=${lng_pair%%-*}
    tgt_lng=${lng_pair##*-}

    echo "apply spm ..."
    python3 $SPM_ENCODE \
        --model $SPM_MODEL \
        --inputs  $PARA_DIR/clean.auth.$lng_pair.$src_lng $PARA_DIR/clean.auth.$lng_pair.$tgt_lng \
        --outputs $PARA_DIR/spm.auth.$lng_pair.$src_lng   $PARA_DIR/spm.auth.$lng_pair.$tgt_lng

    echo "binarize ..."
    fairseq-preprocess \
        --source-lang $src_lng --target-lang $tgt_lng \
        --trainpref $PARA_DIR/spm.auth.$lng_pair \
        --thresholdsrc 0 --thresholdtgt 0 \
        --destdir $PARA_DIR \
        --srcdict $DICT --tgtdict $DICT \
        --workers $N_WORKERS
    
    mv $PARA_DIR/train.$lng_pair.$src_lng.bin $PARA_DIR/train.auth.$lng_pair.$src_lng.bin
    mv $PARA_DIR/train.$lng_pair.$src_lng.idx $PARA_DIR/train.auth.$lng_pair.$src_lng.idx
    mv $PARA_DIR/train.$lng_pair.$tgt_lng.bin $PARA_DIR/train.auth.$lng_pair.$tgt_lng.bin
    mv $PARA_DIR/train.$lng_pair.$tgt_lng.idx $PARA_DIR/train.auth.$lng_pair.$tgt_lng.idx
done


echo "processing authentic+synthetic data (only en-liv)"

echo "concat authentic and synthetic data ..."
cat $PARA_DIR/clean.auth.en-liv.en  $PARA_DIR/clean.syn.en-liv.en  > $PARA_DIR/clean.auth-syn.en-liv.en
cat $PARA_DIR/clean.auth.en-liv.liv $PARA_DIR/clean.syn.en-liv.liv > $PARA_DIR/clean.auth-syn.en-liv.liv

echo "apply spm ..."
python3 $SPM_ENCODE \
    --model $SPM_MODEL \
    --inputs  $PARA_DIR/clean.auth-syn.en-liv.en $PARA_DIR/clean.auth-syn.en-liv.liv \
    --outputs $PARA_DIR/spm.auth-syn.en-liv.en   $PARA_DIR/spm.auth-syn.en-liv.liv

echo "binarize ..."
fairseq-preprocess \
    --source-lang en --target-lang liv \
    --trainpref $PARA_DIR/spm.auth-syn.en-liv \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir $PARA_DIR \
    --srcdict $DICT --tgtdict $DICT \
    --workers $N_WORKERS

mv $PARA_DIR/train.en-liv.en.bin  $PARA_DIR/train.auth-syn.en-liv.en.bin
mv $PARA_DIR/train.en-liv.en.idx  $PARA_DIR/train.auth-syn.en-liv.en.idx
mv $PARA_DIR/train.en-liv.liv.bin $PARA_DIR/train.auth-syn.en-liv.liv.bin
mv $PARA_DIR/train.en-liv.liv.idx $PARA_DIR/train.auth-syn.en-liv.liv.idx