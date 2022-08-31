# WMT22-En-Liv

This is the implementaion of Tencent AI Lab - Shanghai Jiao Tong University (TAL-SJTU) 's En-Liv submission for the [Sixth Conference on Machine Translation (WMT22)](http://www.statmt.org/wmt22/).

<p align="center">
<img src="imgs/training-process.png" alt="training-process"  width="500" />
</p>

## Overview

* **Cross-model word embedding alignment**: transfer the word embeddings of Liv4ever-MT to M2M100, enabling it to support Livonian.
* **4-lingual M2M training**: many-to-many translation training for all language pairs in {En, Liv, Et, Lv}, using only parallel data.
* **Synthetic data generation**: generate synthetic bi-text for En-Liv, using Et and Lv as pivot languages.
* **Combine data and retrain**: combine all the authentic and synthetic bi-text and retrain the model.
* **Fine-tune & post-process**: fine-tune the model on En⇔Liv using the validation set and perform online back-translation using mono-lingual data. Finally, apply rule-based post-processing to the model output.

## Preparation

