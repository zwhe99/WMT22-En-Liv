# WMT22-En-Liv

This is the implementaion of Tencent AI Lab - Shanghai Jiao Tong University (TAL-SJTU) 's En-Liv submissions for the [Sixth Conference on Machine Translation (WMT22)](http://www.statmt.org/wmt22/).


## Overview

<p align="center">
<img src="imgs/training-process.png" alt="training-process"  width="500" />
</p>

* **Cross-model word embedding alignment**: transfer the word embeddings of Liv4ever-MT to M2M100, enabling it to support Livonian.
* **4-lingual M2M training**: many-to-many translation training for all language pairs in {En, Liv, Et, Lv}, using only parallel data.
* **Synthetic data generation**: generate synthetic bi-text for En-Liv, using Et and Lv as pivot languages.
* **Combine data and retrain**: combine all the authentic and synthetic bi-text and retrain the model.
* **Fine-tune & post-process**: fine-tune the model on En⇔Liv using the validation set and perform online back-translation using mono-lingual data. Finally, apply rule-based post-processing to the model output.



## Preparation

#### Download pre-trained models

```bash
# M2M100 1.2B
mkdir -p PTModels/M2M100
wget  -P PTModels/M2M100 https://dl.fbaipublicfiles.com/m2m_100/1.2B_last_checkpoint.pt
wget  -P PTModels/M2M100 https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
wget  -P PTModels/M2M100 https://dl.fbaipublicfiles.com/m2m_100/language_pairs_small_models.txt

# Liv4ever-MT
yum install git-lfs
git lfs install
git clone https://huggingface.co/tartuNLP/liv4ever-mt PTModels/Liv4ever-MT
```



#### Dependencies

* python==3.8.12

* pytorch==1.10.0

* sentencepiece==0.1.96

* fairseq

  ```shell
  pip3 install -e ./fairseq
  ```



## Data

#### Download

We provide filtered data for download, both authentic and synthetic (En-Liv only):

* [Parallel data](https://drive.google.com/drive/folders/1EPS8vcrLTgUDkUT59ddvUFPCoyTj8cXB?usp=sharing)
* [Monolingual data](https://drive.google.com/drive/folders/14ReDmby6y-LUgf2hkh2C8WW_wf-OK-Gm?usp=sharing)

Download the files to the `data/mono` or `data/para` directory, and the structure should be:

```
data
├── data-bin
├── eval
│   ├── benchmark-test.en
│   ├── benchmark-test.et
│   ├── benchmark-test.liv
│   ├── benchmark-test.lv
│   ├── process-eval-data.sh
│   └── wmttest2022.en-de.en
├── mono
│   ├── clean.en
│   ├── clean.liv
│   └── process-mono-data.sh
└── para
    ├── clean.auth.en-et.en
    ├── clean.auth.en-et.et
    ├── clean.auth.en-liv.en
    ├── clean.auth.en-liv.liv
    ├── clean.auth.en-lv.en
    ├── clean.auth.en-lv.lv
    ├── clean.auth.et-liv.et
    ├── clean.auth.et-liv.liv
    ├── clean.auth.et-lv.et
    ├── clean.auth.et-lv.lv
    ├── clean.auth.liv-lv.liv
    ├── clean.auth.liv-lv.lv
    ├── clean.syn.en-liv.en
    ├── clean.syn.en-liv.liv
    └── process-para-data.sh
```



#### Processing

Encode raw text into sentence pieces and binarize (this may take a long time):

```shell
# apply spm and binarize
sh data/eval/process-eval-data.sh
sh data/para/process-para-data.sh
sh data/mono/process-mono-data.sh

# create data-bins
sh data/data/mono/create-data-bin.sh
```

The binary files will be stored in `data/data-bin/auth` (authentic) and `data/data-bin/auth-syn` (authentic+synthetic). 



## Cross-model word embedding alignment (CMEA)

***Note:** You can use `--help` to see the full uage of each script.*

```shell
SRC_MODEL_NAME=liv4ever_mt
TGT_MODEL_NAME=m2m100_1_2B
CEMA_DIR=PTModels/M2M100-CMEA

mkdir -p $CEMA_DIR

# Obtain the overlapping vocabulary
python3 tools/get-overlap.py \
	--d1 PTModels/Liv4ever-MT/dict.src.txt \
	--d2 PTModels/M2M100/model_dict.128k.txt \
	> $CEMA_DIR/overlap-voc.$SRC_MODEL_NAME-$TGT_MODEL_NAME.txt

# Extract word embeddings from models
python3 tools/extract-word-emb.py \
    --model PTModels/Liv4ever-MT/checkpoint_best.pt \
    --dict  PTModels/Liv4ever-MT/dict.src.txt \
    --name $SRC_MODEL_NAME \
    --dest $CEMA_DIR/word-emb-$SRC_MODEL_NAME.pth

python3 tools/extract-word-emb.py \
    --model PTModels/M2M100/1.2B_last_checkpoint.pt \
    --dict  PTModels/M2M100/model_dict.128k.txt \
    --name $TGT_MODEL_NAME \
    --dest $CEMA_DIR/word-emb-$TGT_MODEL_NAME.pth

# Cross-model word embedding alignment
python3 tools/CMEA/supervised-inconsistent-dimensions.py \
    --exp_path $CEMA_DIR \
    --exp_name $SRC_MODEL_NAME-$TGT_MODEL_NAME-cema \
    --exp_id main \
    --src_lang $SRC_MODEL_NAME \
    --tgt_lang $TGT_MODEL_NAME \
    --src_emb_dim 512 \
    --tgt_emb_dim 1024 \
    --n_refinement 0 \
    --cuda False \
    --dico_train $CEMA_DIR/overlap-voc.$SRC_MODEL_NAME-$TGT_MODEL_NAME.txt \
    --src_emb $CEMA_DIR/word-emb-$SRC_MODEL_NAME.pth \
    --tgt_emb $CEMA_DIR/word-emb-$TGT_MODEL_NAME.pth \
    --export pth

# Get the final dictionary (Liv4ever-MT's dict + Lang tokens + madeupwords)
cat PTModels/Liv4ever-MT/dict.trg.txt > $CEMA_DIR/merge_dict.txt
echo "__liv__ 1" >> $CEMA_DIR/merge_dict.txt
echo "__en__ 1"  >> $CEMA_DIR/merge_dict.txt
echo "__et__ 1"  >> $CEMA_DIR/merge_dict.txt
echo "__lv__ 1"  >> $CEMA_DIR/merge_dict.txt
echo "madeupwordforbt 1" >> $CEMA_DIR/merge_dict.txt
echo "madeupword0000 0"  >> $CEMA_DIR/merge_dict.txt
echo "madeupword0001 0"  >> $CEMA_DIR/merge_dict.txt

# Replace the original embedding with the new one
python3 tools/CMEA/change-emb.py \
    --model PTModels/M2M100/1.2B_last_checkpoint.pt \
    --emb1 $CEMA_DIR/$SRC_MODEL_NAME-$TGT_MODEL_NAME-cema/main/vectors-$SRC_MODEL_NAME.pth \
    --emb2 $CEMA_DIR/$SRC_MODEL_NAME-$TGT_MODEL_NAME-cema/main/vectors-$TGT_MODEL_NAME.pth \
    --dict $CEMA_DIR/merge_dict.txt \
    --add-mask \
    --dest $CEMA_DIR/1.2B_last_checkpoint_cmea_emb.pt
```

