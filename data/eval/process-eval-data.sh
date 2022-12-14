#!/bin/sh

set -u

SCRIPT=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
MAIN_DIR=$(realpath "$SCRIPT_DIR/../..")

N_WORKERS=16

# script to normalize unicode
NORM_UNICODE=$MAIN_DIR/tools/corpus-tools/norm_unicode.py

# evaluation data dir
EVAL_DIR=$MAIN_DIR/data/eval

# spm
SPM_MODEL=$MAIN_DIR/PTModels/Liv4ever-MT/enlvetli.model
SPM_ENCODE=$MAIN_DIR/fairseq/scripts/spm_encode.py
DICT=$MAIN_DIR/PTModels/M2M100-CMEA/merge_dict.txt

echo "processing evaluation data (liv4ever-test)"
lng_pairs=(en-et en-liv en-lv et-liv et-lv liv-lv)
for lng_pair in ${lng_pairs[*]}
do
    echo "processing $lng_pair..."
    src_lng=${lng_pair%%-*}
    tgt_lng=${lng_pair##*-}

    echo "apply spm ..."
    python3 $SPM_ENCODE \
        --model $SPM_MODEL \
        --inputs  $EVAL_DIR/benchmark-test.$src_lng     $EVAL_DIR/benchmark-test.$tgt_lng \
        --outputs $EVAL_DIR/benchmark-test.spm.$lng_pair.$src_lng $EVAL_DIR/benchmark-test.spm.$lng_pair.$tgt_lng

    echo "binarize ..."
    fairseq-preprocess \
        --source-lang $src_lng --target-lang $tgt_lng \
        --validpref $EVAL_DIR/benchmark-test.spm.$lng_pair \
        --thresholdsrc 0 --thresholdtgt 0 \
        --destdir $EVAL_DIR \
        --srcdict $DICT --tgtdict $DICT \
        --workers $N_WORKERS
done

echo "noramlizing the references to NFKC"
for lng in en et liv lv
do
    cat benchmark-test.$lng | python3 $NORM_UNICODE "NFKC" > benchmark-test.nfkc.$lng
done


echo "processing evaluation data (round-trip)"
python3 $SPM_ENCODE \
    --model $SPM_MODEL \
    --inputs  $EVAL_DIR/wmttest2022.en-de.en \
    --outputs $EVAL_DIR/wmttest2022.spm.en-de.en

echo "processing evaluation data (wmt22 test)"
echo "apply spm ..."
python3 $SPM_ENCODE \
    --model $SPM_MODEL \
    --inputs  $EVAL_DIR/wmttest2022.en-liv.en  \
    --outputs $EVAL_DIR/wmttest2022.spm.en-liv.en

python3 $SPM_ENCODE \
    --model $SPM_MODEL \
    --inputs  $EVAL_DIR/wmttest2022.liv-en.liv  \
    --outputs $EVAL_DIR/wmttest2022.spm.liv-en.liv

echo "binarize ..."
fairseq-preprocess \
    --source-lang en \
    --testpref $EVAL_DIR/wmttest2022.spm.en-liv \
    --thresholdsrc 0 \
    --destdir $EVAL_DIR \
    --srcdict $DICT  \
    --only-source \
    --workers $N_WORKERS

fairseq-preprocess \
    --source-lang liv \
    --testpref $EVAL_DIR/wmttest2022.spm.liv-en \
    --thresholdsrc 0 \
    --destdir $EVAL_DIR \
    --srcdict $DICT  \
    --only-source \
    --workers $N_WORKERS