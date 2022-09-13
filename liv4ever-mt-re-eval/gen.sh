#!/bin/sh

set -u

SCRIPT=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
MAIN_DIR=$(realpath "$SCRIPT_DIR/..")

# scripts
INTERACTIVE=$MAIN_DIR/fairseq/fairseq_cli/interactive.py
ADD_TAG=$MAIN_DIR/tools/corpus-tools/add_tag.py
SPM_ENCODE=$MAIN_DIR/fairseq/scripts/spm_encode.py

# model
MODEL_DIR=$MAIN_DIR/PTModels/Liv4ever-MT
MODEL_PATH=$MODEL_DIR/checkpoint_best.pt
SPM_MODEL=$MODEL_DIR/enlvetli.model
SRC_DICT=$MODEL_DIR/dict.src.txt
TGT_DICT=$MODEL_DIR/dict.trg.txt

# data
EVAL_DATA_DIR=$MAIN_DIR/data/eval

for src_lng in en et liv lv
do
    for tgt_lng in en et liv lv
    do
        if [[ $src_lng == $tgt_lng ]]
        then
            continue
        fi

        if [[ $tgt_lng == "liv" ]]
        then
            TAG="<2li>"
        else
            TAG="<2${tgt_lng}>"
        fi

        mkdir -p dummy
        cp $SRC_DICT dummy/dict.$src_lng.txt
        cp $TGT_DICT dummy/dict.$tgt_lng.txt

        cat $EVAL_DATA_DIR/benchmark-test.$src_lng | \
        python3 $SPM_ENCODE --model $SPM_MODEL | \
        python3 $ADD_TAG $TAG | \
        python3 $INTERACTIVE dummy \
            --batch-size 100 \
            --buffer-size 1024 \
            --path $MODEL_PATH \
            -s $src_lng  -t $tgt_lng \
            --remove-bpe 'sentencepiece' \
            --beam 5 > $SCRIPT_DIR/${src_lng}-${tgt_lng}.gen
        cat $SCRIPT_DIR/${src_lng}-${tgt_lng}.gen | grep -P "^H" | sort -V | cut -f 3-  > $SCRIPT_DIR/${src_lng}-${tgt_lng}.hyp
        rm -r dummy
    done
done



