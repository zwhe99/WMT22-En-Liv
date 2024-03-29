# WMT22-En-Liv

This is the implementaion of Tencent AI Lab - Shanghai Jiao Tong University (TAL-SJTU) 's English-Livonian submissions for the [Sixth Conference on Machine Translation (WMT22)](http://www.statmt.org/wmt22/). We provide all the [models](https://drive.google.com/drive/folders/1o5Ux-mGOARivmqlCube7w0vRjeDjsMBr?usp=sharing), [data](https://drive.google.com/drive/folders/164iYBFG6-A-2YBRYC9sppfarehRez-Bi?usp=sharing), code and scripts in this repository. More details are available in our system description [paper](https://arxiv.org/abs/2210.08742).

**Note:** We find that Liv4ever-MT has been underestimated due to inconsistent Unicode normalization. Please see [liv4ever-mt-re-eval](liv4ever-mt-re-eval) to reproduce our results.

**News**

* We won **1st** place🥇for English=>Livonian and **2nd** place🥈for Livonian<=English (Unconstrained System). [[Official Results]](https://statmt.org/wmt22/pdf/2022.wmt-1.1.pdf)


## Overview

<p align="center">
<img src="imgs/training-process.png" alt="training-process"  width="500" />
</p>


* **Cross-model word embedding alignment**: transfer the word embeddings of Liv4ever-MT to M2M100, enabling it to support Livonian.
* **4-lingual M2M training**: many-to-many translation training for all language pairs in {En, Liv, Et, Lv}, using only parallel data.
* **Synthetic data generation**: generate synthetic bi-text for En-Liv, using Et and Lv as pivot languages.
* **Combine data and retrain**: combine all the authentic and synthetic bi-text and retrain the model.
* **Fine-tune & post-process**: fine-tune the model on En⇔Liv using the validation set and perform online back-translation using monolingual data. Finally, apply rule-based post-processing to the model output.



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

* nltk==3.7

* sacrebleu=2.0.0

* fairseq

  ```shell
  pip3 install -e ./fairseq
  ```



## Cross-model word embedding alignment (CMEA)

* Processed model: [1.2B_last_checkpoint_cmea_emb.pt](https://drive.google.com/file/d/1ZDZuVtk4wuFlhXQigvvbDQCko5CKaiu9/view?usp=sharing)

* Dictionary: [merge_dict.txt](https://drive.google.com/file/d/1-MMBtu9kYkJcRYAKiP65GLmVFCc1APyS/view?usp=sharing)

* CMEA scripts

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
  sed -n '128001,128100p' PTModels/M2M100/model_dict.128k.txt >> $CEMA_DIR/merge_dict.txt
  
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
  
  echo "The processed model is stored in $CEMA_DIR/1.2B_last_checkpoint_cmea_emb.pt"
  echo "The processed dictionary is stored in $CEMA_DIR/merge_dict.txt"
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
sh data/data-bin/create-data-bin.sh
```

The binary files will be stored in `data/data-bin/auth` (authentic) and `data/data-bin/auth-syn` (authentic+synthetic). 



## Model training

**4-lingual M2M training**

* GPUs: 4 nodes x 8 A100-SXM4-40GB/node

* Trained model: [m2m04.pt](https://drive.google.com/file/d/1U0eYttJEzhoAcLyPUSpo2Ol-YcOpI_l9/view?usp=sharing)

* Training script:

  ```shell
  $EXP_NAME=ptm.mm100-1.2b-cmea+task.mt+lang.enlvetli+temp.5+data.auth
  mkdir -p $EXP_NAME
  
  python3 -m torch.distributed.launch --nproc_per_node=8 \
     --nnodes=4 --node_rank=0 --master_addr="xxx.xxx.xxx.xxx" \
     --master_port=xxxxx \
     $(which fairseq-train) data/data-bin/auth \
     --finetune-from-model PTModels/M2M100-CMEA/1.2B_last_checkpoint_cmea_emb.pt \
     --num-workers 0 \
     --encoder-normalize-before  \
     --decoder-normalize-before  \
     --arch transformer_wmt_en_de_big \
     --task multilingual_semisupervised_translation \
     --train-tasks mt \
     --share-all-embeddings  \
     --share-decoder-input-output-embed  \
     --encoder-layerdrop 0.05 \
     --decoder-layerdrop 0.05 \
     --activation-dropout 0.0 \
     --encoder-layers 24 \
     --decoder-layers 24 \
     --encoder-ffn-embed-dim 8192 \
     --decoder-ffn-embed-dim 8192 \
     --encoder-embed-dim 1024 \
     --decoder-embed-dim 1024 \
     --sampling-method temperature \
     --sampling-temperature 5 \
     --encoder-langtok src \
     --decoder-langtok  \
     --langs en,liv,et,lv \
     --lang-pairs en-liv,liv-en,en-et,et-en,en-lv,lv-en,liv-et,et-liv,liv-lv,lv-liv,et-lv,lv-et \
     --criterion label_smoothed_cross_entropy \
     --label-smoothing 0.2 \
     --optimizer adam \
     --adam-eps 1e-08 \
     --adam-betas 0.9,0.98 \
     --lr-scheduler inverse_sqrt \
     --lr 0.0005 \
     --warmup-init-lr 1e-07 \
     --warmup-updates 2000 \
     --max-update 10000 \
     --dropout 0.3 \
     --attention-dropout 0.1 \
     --weight-decay 0.0 \
     --max-tokens 1024 \
     --max-tokens-valid 1024 \
     --update-freq 2 \
     --virtual-epoch-size 10000000 \
     --skip-remainder-batch  \
     --no-progress-bar  \
     --log-format simple \
     --log-interval 2 \
     --best-checkpoint-metric loss \
     --patience 5 \
     --skip-invalid-size-inputs-valid-test  \
     --no-epoch-checkpoints  \
     --eval-lang-pairs et-liv,liv-et,lv-liv,liv-lv \
     --valid-subset valid \
     --validate-interval-updates 500 \
     --save-interval-updates 500 \
     --keep-interval-updates 5 \
     --fp16  \
     --seed 42 \
     --ddp-backend no_c10d \
     --save-dir $EXP_NAME/ckpts \
     --distributed-no-spawn  \
     --tensorboard-logdir $EXP_NAME/tensorboard
  
  mv $EXP_NAME/ckpts/checkpoint_best.pt $EXP_NAME/ckpts/m2m04.pt
  ```
  
  

**Combine data and retrain**

* GPUs: 4 nodes x 8 A100-SXM4-40GB/node

* Trained model: [m2m04-retrained.pt](https://drive.google.com/file/d/1gMFK_1BbE3OGTtMcXnTdxdfdLZ8Vm2bK/view?usp=sharing) (slightly different from that in the paper)

* Training script:

  ```shell
  $EXP_NAME=ptm.mm100-1.2b-cema+task.mt+lang.enlvetli+samp.concat+data.auth-syn
  mkdir -p $EXP_NAME
  
  python3 -m torch.distributed.launch --nproc_per_node=8 \
     --nnodes=4 --node_rank=0 --master_addr="xxx.xxx.xxx.xxx" \
     --master_port=xxxxx \
     $(which fairseq-train) data/data-bin/auth-syn \
     --finetune-from-model PTModels/M2M100-CMEA/1.2B_last_checkpoint_cmea_emb.pt \
     --num-workers 0 \
     --encoder-normalize-before  \
     --decoder-normalize-before  \
     --arch transformer_wmt_en_de_big \
     --task multilingual_semisupervised_translation \
     --train-tasks mt \
     --share-all-embeddings  \
     --share-decoder-input-output-embed  \
     --encoder-layerdrop 0.05 \
     --decoder-layerdrop 0.05 \
     --activation-dropout 0.0 \
     --encoder-layers 24 \
     --decoder-layers 24 \
     --encoder-ffn-embed-dim 8192 \
     --decoder-ffn-embed-dim 8192 \
     --encoder-embed-dim 1024 \
     --decoder-embed-dim 1024 \
     --encoder-langtok src \
     --decoder-langtok  \
     --langs en,liv,et,lv \
     --lang-pairs en-liv,liv-en,en-et,et-en,en-lv,lv-en,liv-et,et-liv,liv-lv,lv-liv,et-lv,lv-et \
     --criterion label_smoothed_cross_entropy \
     --label-smoothing 0.2 \
     --optimizer adam \
     --adam-eps 1e-08 \
     --adam-betas 0.9,0.98 \
     --lr-scheduler inverse_sqrt \
     --lr 0.0005 \
     --warmup-init-lr 1e-07 \
     --warmup-updates 2000 \
     --max-update 10000 \
     --dropout 0.3 \
     --attention-dropout 0.1 \
     --weight-decay 0.0 \
     --max-tokens 1024 \
     --max-tokens-valid 1024 \
     --update-freq 2 \
     --virtual-epoch-size 10000000 \
     --skip-remainder-batch  \
     --no-progress-bar  \
     --log-format simple \
     --log-interval 2 \
     --best-checkpoint-metric loss \
     --patience 10 \
     --skip-invalid-size-inputs-valid-test  \
     --no-epoch-checkpoints  \
     --eval-lang-pairs en-liv,liv-en \
     --valid-subset valid \
     --validate-interval-updates 500 \
     --save-interval-updates 500 \
     --keep-interval-updates 5 \
     --fp16  \
     --seed 42 \
     --ddp-backend no_c10d \
     --save-dir $EXP_NAME/ckpts \
     --distributed-no-spawn  \
     --tensorboard-logdir $EXP_NAME/tensorboard
  
  mv $EXP_NAME/ckpts/checkpoint_best.pt $EXP_NAME/ckpts/m2m04-retrained.pt
  ```

  


**Fintuning**

* GPUs: 1 nodes x 1 A100-SXM4-40GB/node

* Trained model: [m2m04-retrained-finetuned.pt](https://drive.google.com/file/d/1srBY40-uQWFrjpFtCBwLpsJ4Ncfv9olJ/view?usp=sharing) (slightly different from that in the paper)

* Training script:

  ```shell
  $EXP_NAME=ptm.retrained+task.mt-bt+lang.enliv+samp.uni+data.valid-and-mono
  mkdir -p $EXP_NAME
  
  fairseq-train data/data-bin/auth-syn \
     --train-subset finetune \
     --finetune-from-model ptm.mm100-1.2b-cema+task.mt+lang.enlvetli+samp.concat+data.auth-syn/ckpts/m2m04-retrained.pt \
     --num-workers 0 \
     --encoder-normalize-before  \
     --decoder-normalize-before  \
     --arch transformer_wmt_en_de_big \
     --task multilingual_semisupervised_translation \
     --train-tasks mt,bt \
     --share-all-embeddings  \
     --share-decoder-input-output-embed  \
     --encoder-layerdrop 0.05 \
     --decoder-layerdrop 0.05 \
     --activation-dropout 0.0 \
     --encoder-layers 24 \
     --decoder-layers 24 \
     --encoder-ffn-embed-dim 8192 \
     --decoder-ffn-embed-dim 8192 \
     --encoder-embed-dim 1024 \
     --decoder-embed-dim 1024 \
     --sampling-method uniform \
     --encoder-langtok src \
     --decoder-langtok  \
     --langs en,liv,et,lv \
     --lang-pairs liv-en,en-liv \
     --criterion label_smoothed_cross_entropy \
     --label-smoothing 0.2 \
     --optimizer adam \
     --adam-eps 1e-08 \
     --adam-betas 0.9,0.98 \
     --lr-scheduler inverse_sqrt \
     --lr 0.0001 \
     --warmup-init-lr 1e-07 \
     --warmup-updates 2000 \
     --max-update 500 \
     --dropout 0.3 \
     --attention-dropout 0.1 \
     --weight-decay 0.0 \
     --max-tokens 1024 \
     --max-tokens-valid 1024 \
     --update-freq 2 \
     --virtual-epoch-size 10000000 \
     --no-progress-bar  \
     --log-format simple \
     --log-interval 2 \
     --no-epoch-checkpoints  \
     --save-interval-updates 50 \
     --keep-interval-updates 2 \
     --disable-validation  \
     --fp16  \
     --seed 42 \
     --ddp-backend no_c10d \
     --save-dir $EXP_NAME/ckpts \
     --distributed-no-spawn  \
     --tensorboard-logdir $EXP_NAME/tensorboard
  
  mv $EXP_NAME/ckpts/checkpoint_last.pt $EXP_NAME/ckpts/m2m04-retrained-finetuned.pt
  ```



## Evaluation

### Test set

**Generate translations**

```shell
MODEL_PATH=ptm.retrained+task.mt-bt+lang.enliv+samp.uni+data.valid-and-mono/ckpts/m2m04-retrained-finetuned.pt
DICT_PATH=PTModels/M2M100-CMEA/merge_dict.txt
LNG_PAIRS=liv-en,en-liv
LNGS=en,liv,et,lv

for lng_pair in en-liv liv-en
do
    SRC=${lng_pair%%-*}
    TGT=${lng_pair##*-}

    # generate
    fairseq-generate data/data-bin/auth \
        --batch-size 128 \
        --path $MODEL_PATH \
        --fixed-dictionary $DICT_PATH \
        -s $SRC  -t $TGT \
        --remove-bpe 'sentencepiece' \
        --beam 5 \
        --task multilingual_semisupervised_translation \
        --lang-pairs $LNG_PAIRS \
        --langs  $LNGS \
        --decoder-langtok \
        --encoder-langtok src \
        --gen-subset test > wmttest2022.$SRC-$TGT.gen
    cat wmttest2022.$SRC-$TGT.gen | grep -P "^H" | sort -V | cut -f 3-  > wmttest2022.$SRC-$TGT.hyp

    # generate (no-repeat)
    fairseq-generate data/data-bin/auth \
        --batch-size 128 \
        --path $MODEL_PATH \
        --fixed-dictionary $DICT_PATH \
        -s $SRC  -t $TGT \
        --remove-bpe 'sentencepiece' \
        --beam 5 \
        --no-repeat-ngram-size	2 \
        --task multilingual_semisupervised_translation \
        --lang-pairs $LNG_PAIRS \
        --langs  $LNGS \
        --decoder-langtok \
        --encoder-langtok src \
        --gen-subset test > wmttest2022.$SRC-$TGT.no-repeat.gen
    cat wmttest2022.$SRC-$TGT.no-repeat.gen | grep -P "^H" | sort -V | cut -f 3-  > wmttest2022.$SRC-$TGT.no-repeat.hyp
done
```



**Post-processing**

```shell
for lng_pair in en-liv liv-en
do
    SRC=${lng_pair%%-*}
    TGT=${lng_pair##*-}

    if [[ $TGT == "liv" ]]
    then
        python3 tools/post-process.py \
            --src-file data/eval/wmttest2022.$SRC-$TGT.$SRC \
            --hyp-file wmttest2022.$SRC-$TGT.hyp \
            --no-repeat-hyp-file wmttest2022.$SRC-$TGT.no-repeat.hyp \
            --lang $TGT > wmttest2022.$SRC-$TGT.post-processed.hyp
    else
        python3 tools/post-process.py \
            --src-file data/eval/wmttest2022.$SRC-$TGT.$SRC \
            --hyp-file wmttest2022.$SRC-$TGT.hyp \
            --lang $TGT > wmttest2022.$SRC-$TGT.post-processed.hyp
    fi
done
```



**Evaluate**

```shell
echo "Before post-processing:"
cat wmttest2022.en-liv.hyp | sacrebleu data/references/generaltest2022.en-liv.ref.A.liv
cat wmttest2022.liv-en.hyp | sacrebleu data/references/generaltest2022.liv-en.ref.A.en

echo "After post-processing:"
cat wmttest2022.en-liv.post-processed.hyp | sacrebleu data/references/generaltest2022.en-liv.ref.A.liv
cat wmttest2022.liv-en.post-processed.hyp | sacrebleu data/references/generaltest2022.liv-en.ref.A.en
```

Outputs:

```json
Before post-processing:
{
 "name": "BLEU",
 "score": 16.1,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0",
 "verbose_score": "47.0/21.0/10.9/6.2 (BP = 1.000 ratio = 1.050 hyp_len = 9713 ref_len = 9251)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.0.0"
}
{
 "name": "BLEU",
 "score": 30.8,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0",
 "verbose_score": "62.3/37.0/24.0/16.2 (BP = 1.000 ratio = 1.003 hyp_len = 10628 ref_len = 10599)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.0.0"
}

After post-processing:
{
 "name": "BLEU",
 "score": 17.0,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0",
 "verbose_score": "49.7/22.3/11.6/6.6 (BP = 1.000 ratio = 1.010 hyp_len = 9342 ref_len = 9251)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.0.0"
}
{
 "name": "BLEU",
 "score": 30.8,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0",
 "verbose_score": "62.3/37.0/24.0/16.2 (BP = 1.000 ratio = 1.003 hyp_len = 10628 ref_len = 10599)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.0.0"
}
```



### Round-trip BLEU

**Generate translations**

```shell
MODEL_PATH=ptm.retrained+task.mt-bt+lang.enliv+samp.uni+data.valid-and-mono/ckpts/m2m04-retrained-finetuned.pt
DICT_PATH=PTModels/M2M100-CMEA/merge_dict.txt
LNG_PAIRS=liv-en,en-liv
LNGS=en,liv,et,lv

EVAL_DIR=data/eval
SOURCE_FILE=$EVAL_DIR/wmttest2022.en-de.en
SOURCE_SPM_FILE=$EVAL_DIR/wmttest2022.spm.en-de.en

# generate
cat $SOURCE_SPM_FILE | fairseq-interactive $EVAL_DIR \
    --batch-size 128 \
    --buffer-size 1024 \
    --path $MODEL_PATH \
    --fixed-dictionary $DICT_PATH \
    -s en  -t liv \
    --beam 5 \
    --task multilingual_semisupervised_translation \
    --lang-pairs $LNG_PAIRS \
    --langs  $LNGS \
    --decoder-langtok \
    --encoder-langtok src | grep -P "^H" | sort -V | cut -f 3- > round-trip.spm.en-liv
    
cat round-trip.spm.en-liv | fairseq-interactive $EVAL_DIR \
    --batch-size 128 \
    --buffer-size 1024 \
    --path $MODEL_PATH \
    --fixed-dictionary $DICT_PATH \
    --remove-bpe 'sentencepiece' \
    -s liv  -t en \
    --beam 5 \
    --task multilingual_semisupervised_translation \
    --lang-pairs $LNG_PAIRS \
    --langs  $LNGS \
    --decoder-langtok \
    --encoder-langtok src | grep -P "^H" | sort -V | cut -f 3- > round-trip.en-liv-en

```



**Evaluate**

```shell
cat round-trip.en-liv-en | sacrebleu $SOURCE_FILE
```

Outputs:

```json
{
 "name": "BLEU",
 "score": 36.8,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0",
 "verbose_score": "69.9/45.6/31.7/22.4 (BP = 0.950 ratio = 0.951 hyp_len = 337570 ref_len = 354789)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.0.0"
}
```



### Citation

Please cite our system description [paper](https://arxiv.org/abs/2210.08742) if you found the resources in this repository useful.

```
@inproceedings{he-etal-2022-tencent,
    title = "Tencent {AI} Lab - Shanghai Jiao Tong University Low-Resource Translation System for the {WMT}22 Translation Task",
    author = "He, Zhiwei  and
      Wang, Xing  and
      Tu, Zhaopeng  and
      Shi, Shuming  and
      Wang, Rui",
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.18",
    pages = "260--267",
}
```
