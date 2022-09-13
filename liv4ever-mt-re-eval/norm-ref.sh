#!/bin/sh

set -u

SCRIPT=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
MAIN_DIR=$(realpath "$SCRIPT_DIR/..")

# data
EVAL_DATA_DIR=$MAIN_DIR/data/eval

# script
NORM_UNICODE=$MAIN_DIR/tools/corpus-tools/norm_unicode.py

for lng in en et lv liv
do
    cat $EVAL_DATA_DIR/benchmark-test.$lng | python3 $NORM_UNICODE "NFKC" > $EVAL_DATA_DIR/benchmark-test.nfkc.$lng
done