#!/bin/sh

set -u

SCRIPT=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
MAIN_DIR=$(realpath "$SCRIPT_DIR/../..")

DATA_BIN_DIR=$MAIN_DIR/data/data-bin
PARA_DIR=$MAIN_DIR/data/para
MONO_DIR=$MAIN_DIR/data/mono
EVAL_DIR=$MAIN_DIR/data/eval

for data_type in auth auth-syn
do
    DEST_DIR=$DATA_BIN_DIR/$data_type
    mkdir -p $DEST_DIR

    # para
    lng_pairs=(en-et en-liv en-lv et-liv et-lv liv-lv)
    for lng_pair in ${lng_pairs[*]}
    do
        src_lng=${lng_pair%%-*}
        tgt_lng=${lng_pair##*-}

        if [[ $data_type == "auth-syn" && $lng_pair == "en-liv" ]]
        then
            ln -sf $PARA_DIR/train.auth-syn.$lng_pair.$src_lng.bin $DEST_DIR/train.$lng_pair.$src_lng.bin
            ln -sf $PARA_DIR/train.auth-syn.$lng_pair.$src_lng.idx $DEST_DIR/train.$lng_pair.$src_lng.idx
            ln -sf $PARA_DIR/train.auth-syn.$lng_pair.$tgt_lng.bin $DEST_DIR/train.$lng_pair.$tgt_lng.bin
            ln -sf $PARA_DIR/train.auth-syn.$lng_pair.$tgt_lng.idx $DEST_DIR/train.$lng_pair.$tgt_lng.idx
        else
            ln -sf $PARA_DIR/train.auth.$lng_pair.$src_lng.bin     $DEST_DIR/train.$lng_pair.$src_lng.bin
            ln -sf $PARA_DIR/train.auth.$lng_pair.$src_lng.idx     $DEST_DIR/train.$lng_pair.$src_lng.idx
            ln -sf $PARA_DIR/train.auth.$lng_pair.$tgt_lng.bin     $DEST_DIR/train.$lng_pair.$tgt_lng.bin
            ln -sf $PARA_DIR/train.auth.$lng_pair.$tgt_lng.idx     $DEST_DIR/train.$lng_pair.$tgt_lng.idx
        fi
    done

    # mono
    cp -rs $MONO_DIR/*.bin $DEST_DIR
    cp -rs $MONO_DIR/*.idx $DEST_DIR

    # dict
    cp -rs $PARA_DIR/dict* $DEST_DIR

    # eval
    cp -rs $EVAL_DIR/*.bin $DEST_DIR
    cp -rs $EVAL_DIR/*.idx $DEST_DIR
    ln -sf $DEST_DIR/test.en-None.en.bin $DEST_DIR/test.en-liv.en.bin 
    ln -sf $DEST_DIR/test.en-None.en.idx $DEST_DIR/test.en-liv.en.idx 
    ln -sf $DEST_DIR/test.liv-None.liv.bin $DEST_DIR/test.liv-en.liv.bin 
    ln -sf $DEST_DIR/test.liv-None.liv.idx $DEST_DIR/test.liv-en.liv.idx 

    # data for fine-tuning
    ln -sf $DEST_DIR/valid.en-liv.en.bin    $DEST_DIR/finetune.en-liv.en.bin
    ln -sf $DEST_DIR/valid.en-liv.en.idx    $DEST_DIR/finetune.en-liv.en.idx
    ln -sf $DEST_DIR/valid.en-liv.liv.bin   $DEST_DIR/finetune.en-liv.liv.bin
    ln -sf $DEST_DIR/valid.en-liv.liv.idx   $DEST_DIR/finetune.en-liv.liv.idx

    ln -sf $DEST_DIR/train.en-None.en.bin   $DEST_DIR/finetune.en-None.en.bin
    ln -sf $DEST_DIR/train.en-None.en.idx   $DEST_DIR/finetune.en-None.en.idx
    ln -sf $DEST_DIR/train.liv-None.liv.bin $DEST_DIR/finetune.liv-None.liv.bin
    ln -sf $DEST_DIR/train.liv-None.liv.idx $DEST_DIR/finetune.liv-None.liv.idx
done