#!/bin/sh

set -u

SCRIPT=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
MAIN_DIR=$(realpath "$SCRIPT_DIR/../..")

N_WORKERS=16

# monolingual data dir
MONO_DIR=$MAIN_DIR/data/mono

# spm
SPM_MODEL=$MAIN_DIR/PTModels/Liv4ever-MT/enlvetli.model
SPM_ENCODE=$MAIN_DIR/fairseq/scripts/spm_encode.py
DICT=$MAIN_DIR/PTModels/M2M100-CMEA/merge_dict.txt

for lng in en liv
do
    echo "processing $lng ..."
    echo "apply spm ..."
    python3 $SPM_ENCODE \
        --model   $SPM_MODEL \
        --inputs  $MONO_DIR/clean.$lng \
        --outputs $MONO_DIR/spm.$lng

    echo "binarize ..."
    fairseq-preprocess \
        --source-lang $lng \
        --trainpref $MONO_DIR/spm \
        --thresholdsrc 0 --thresholdtgt 0 \
        --destdir $MONO_DIR \
        --srcdict $DICT \
        --only-source \
        --workers $N_WORKERS
done