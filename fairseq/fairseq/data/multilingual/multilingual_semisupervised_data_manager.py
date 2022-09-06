import itertools
import logging
import os
import torch
import numpy as np
from collections import OrderedDict, defaultdict
from fairseq import utils
from fairseq.data import (
    SampledMultiEpochDataset,
    BacktranslationDataset,
    RoundRobinZipDatasets,
    ConcatDataset,
    LanguagePairDataset,
    data_utils
)
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager, 
    load_sampling_weights, 
    _lang_id
)
from fairseq.data.multilingual.multilingual_utils import get_lang_tok
from fairseq.data.multilingual.sampled_multi_dataset import CollateFormat

logger = logging.getLogger(__name__)

MT_NAME='mt'
BT_NAME="bt"
DAE_NAME="dae"
MASK_TOKEN="<mask>"

class WrapperRoundRobinZipDatasets(RoundRobinZipDatasets):
    """
    We need to modify RoundRobinZipDatasets so that 
    it can contain SampledMultiEpochDataset
    """
    def __init__(self, datasets, eval_key=None):
        super().__init__(datasets, eval_key)
    
    def refresh(self):
        self.longest_dataset_key = max(self.datasets, key=lambda k: len(self.datasets[k]))
        self.longest_dataset = self.datasets[self.longest_dataset_key]
        self.ordered_indices()
    
    def set_epoch(self, epoch):
        """Set epoch for each dataset

        Args:
            epoch (int): epoch num
        """
        for _, ds in self.datasets.items():
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def filter_indices_by_size(self, indices, max_positions=None):
        """
        Filter each sub-dataset independently, then update the round robin to work
        on the filtered sub-datasets.
        """

        if not isinstance(max_positions, dict):
            max_positions = {k: max_positions for k in self.datasets.keys()}
        ignored_some = False
        
        for key, dataset in self.datasets.items():
            self._ordered_indices[key], ignored = dataset.filter_indices_by_size(
                self._ordered_indices[key], max_positions[key]
            )
            if len(ignored) > 0:
                ignored_some = True
                logger.warning(
                    f"[{key}] {len(ignored)} samples from {key} have invalid sizes and will be skipped, "
                    f"max_positions={max_positions[key]}, first few sample ids={ignored[:10]}"
                )
        # Since we are modifying in place the _ordered_indices,
        # it's not possible anymore to return valid ignored indices.
        # Hopefully the extra debug information print above should be enough to debug.
        # Ideally we would receive ignore_invalid_inputs so that we could have
        # a proper error message.
        return (np.arange(len(self)), [0] if ignored_some else [])

    def ordered_indices(self):
        """Ordered indices for batching."""
        # Call the underlying dataset's ordered_indices() here, so that we
        # get the same random ordering as we would have from using the
        # underlying sub-datasets directly.
        self._ordered_indices = OrderedDict(
            [
                (key, dataset.ordered_indices())
                for key, dataset in self.datasets.items()
            ]
        )
        return np.arange(len(self))
class WrapperBacktranslationDataset(BacktranslationDataset):
    """
    We need to modify the BacktranslationDataset to give it the 
    self.sizes property. 
    
    NOTE: we use *tgt_dataset* to approximate the size of the 
    source dataset, since we do not know the actual size until 
    after backtranslation.
    """

    def __init__(
        self,
        tgt_dataset,
        src_dict,
        tgt_dict=None,
        backtranslation_fn=None,
        output_collater=None,
        cuda=True,
        **kwargs
    ):
        super().__init__(
            tgt_dataset,
            src_dict,
            tgt_dict,
            backtranslation_fn,
            output_collater,
            cuda,
            **kwargs
        )
        self.sizes = (
            np.vstack((self.tgt_dataset.sizes, self.tgt_dataset.sizes)).T
        )

class DenoisingDataset(WrapperBacktranslationDataset):
    def __init__(
        self,
        max_word_shuffle_distance,
        word_dropout_prob,
        word_blanking_prob,
        tgt_dataset,
        src_dict,
        lang_tok_id,
        tgt_dict=None,
        backtranslation_fn=None,
        output_collater=None,
        cuda=False,
        seed=42,
        **kwargs
    ):  
        assert backtranslation_fn == None, "DenoisingDatatset has set backtranslation_fn in __init__ function."
        self.max_word_shuffle_distance = max_word_shuffle_distance
        self.word_dropout_prob = word_dropout_prob
        self.word_blanking_prob = word_blanking_prob
        self.lang_tok_id = lang_tok_id
        self.mask_id = src_dict
        self.seed = seed

        def add_noise(samples):
            with data_utils.numpy_seed(self.seed + samples["id"][0].item()):
                src_tokens = torch.flip(torch.t(samples["net_input"]["src_tokens"]), [0]).cpu()
                src_lengths = samples["net_input"]["src_lengths"].cpu()
                src_tokens, src_lengths = self.word_shuffle(src_tokens, src_lengths)
                src_tokens, src_lengths = self.word_dropout(src_tokens, src_lengths)
                src_tokens, src_lengths = self.word_blank(src_tokens, src_lengths)

                return [[{
                    "tokens": torch.flip(tk[:l], [0])
                }] for tk, l in zip(src_tokens.T, src_lengths)]
            

        super().__init__(
            tgt_dataset,
            src_dict,
            tgt_dict,
            add_noise,
            output_collater,
            cuda,
            **kwargs
        )
    
    def word_shuffle(self, x, l):
        """
        Randomly shuffle input words.
        """
        if self.max_word_shuffle_distance == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.max_word_shuffle_distance, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        assert self.max_word_shuffle_distance > 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = np.arange(l[i] - 1) + noise[:l[i] - 1, i]
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l

    def word_dropout(self, x, l):
        """
        Randomly drop input words.
        """
        if self.word_dropout_prob == 0:
            return x, l
        assert 0 < self.word_dropout_prob < 1

        # define words to drop
        bos = self.src_dict.eos_index
        eos = self.lang_tok_id
        if not (x[0] == bos).sum() == l.size(0):
            logger.error(f"x[0]: {x[0]}")
            logger.error(f"bos: {bos}")
        assert (x[0] == bos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.word_dropout_prob
        keep[0] = 1  # do not drop the start sentence symbol

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[j, i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(eos)
            assert len(new_s) >= 3 and new_s[0] == bos and new_s[-1] == eos
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.src_dict.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        
        return x2, l2

    def word_blank(self, x, l):
        """
        Randomly blank input words.
        """
        if self.word_blanking_prob == 0:
            return x, l
        assert 0 < self.word_blanking_prob < 1

        # define words to blank
        bos = self.src_dict.eos_index
        eos = self.lang_tok_id
        assert (x[0] == bos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.word_blanking_prob
        keep[0] = 1  # do not blank the start sentence symbol

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[j, i] else self.src_dict.index(MASK_TOKEN) for j, w in enumerate(words)]
            new_s.append(eos)
            assert len(new_s) == l[i] and new_s[0] == bos and new_s[-1] == eos
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.src_dict.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l



class MultilingualSemisupervisedDatasetManager(MultilingualDatasetManager):
    def __init__(self, args, lang_pairs, eval_lang_pairs, langs, dicts, sampling_method, training):
        super().__init__(args, lang_pairs, langs, dicts, sampling_method)

        for v in self.dicts.values():
            v.add_symbol(MASK_TOKEN)

        self._training_data_sizes = {
            BT_NAME: defaultdict(lambda: {}),
            DAE_NAME: defaultdict(lambda: {}),
            MT_NAME: defaultdict(lambda: {})
        }
        if eval_lang_pairs is not None:
            self.eval_lang_pairs = eval_lang_pairs
        else:
            self.eval_lang_pairs = lang_pairs

        if DAE_NAME in args.train_tasks or BT_NAME in args.train_tasks:
            self.training_mono_langs = sorted(list(
                set([lp.split('-')[0] for lp in lang_pairs] + 
                    [lp.split('-')[1] for lp in lang_pairs]
                )
            ))
        else:
            self.training_mono_langs = []

        self.tgt_lang_mask_lang_tokens_idx = {}
        self.tgt_lang_mask_words_idx = {}
        for lang_pair in self.lang_pairs:
            tgt_lang = lang_pair.split('-')[-1].strip()
            self.tgt_lang_mask_lang_tokens_idx[tgt_lang] = [
                self.get_langtok_index(get_lang_tok(lang, args.lang_tok_style), self.dicts[tgt_lang]) for lang in self.langs
            ]
            self.tgt_lang_mask_words_idx[tgt_lang] = self.tgt_lang_mask_lang_tokens_idx[tgt_lang]


        if training:
            self.only_target_token_percent = args.only_target_token_percent
            self.only_target_token_before_step = args.only_target_token_before_step

            if self.only_target_token_percent != 1 and self.only_target_token_before_step != 0:
                logger.info(
                    f"In the first {self.only_target_token_before_step} steps, "
                    f"on-the-fly BT only generates the first {self.only_target_token_percent*100}% of tokens in the target language."
                )
                self.if_mask = True
                
                for lang_pair in self.lang_pairs:
                    tgt_lang = lang_pair.split('-')[-1].strip()
                    tgt_lang_dict_file = os.path.join(args.data, "mono_dicts", f"dict.{tgt_lang}.txt")
                    with open(tgt_lang_dict_file, 'r', encoding='utf-8') as f:
                        word2freq = {}
                        zero_cnt = 0
                        non_zero_cnt = 0
                        for line in f:
                            word, freq = line.split()
                            freq = float(freq)
                            word2freq[word] = freq
                            if freq == 0:
                                zero_cnt += 1
                            else:
                                non_zero_cnt += 1
                        mask_cnt = zero_cnt + int(non_zero_cnt * (1 - self.only_target_token_percent))
                        self.tgt_lang_mask_words_idx[tgt_lang] += [
                                                                self.dicts[tgt_lang].index(w)
                                                                for w, _ in 
                                                                    sorted(word2freq.items(), key=lambda item: item[1])
                                                            ][ :mask_cnt]
                        logger.info(
                            f"{tgt_lang}: Mask {len(self.tgt_lang_mask_words_idx[tgt_lang])} tokens."
                        )

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument(
            "--bt-sampling-weights-from-file",
            help='a file contain a python dictionary of how to sample back-translation data sets, \
                                e.g. { "main:en-de": 0.2, "main:de-en": 0.2, \
                                        "main:en-fr: 0.3, "main:fr-en": 0.3 }',
            default=None,
            type=str,
        )
        parser.add_argument(
            "--bt-sampling-weights",
            help='a dictionary of how to sample back-translation data sets, \
                                e.g. { "main:en-de": 0.2, "main:de-en": 0.2, \
                                        "main:en-fr: 0.3, "main:fr-en": 0.3 }',
            default=None,
            type=lambda uf: utils.eval_str_dict(uf, type=str),
        )
        parser.add_argument(
            "--dae-sampling-weights-from-file",
            help='a file contain a python dictionary of how to sample denoising auto-encoding data sets, \
                                e.g. { "main:en-en": 0.2, "main:de-de": 0.2, \
                                        "main:fr-fr: 0.3, "main:zh-zh": 0.3 }',
            default=None,
            type=str,
        )
        parser.add_argument(
            "--dae-sampling-weights",
            help='a dictionary of how to sample denoising auto-encoding data sets, \
                                e.g. { "main:en-en": 0.2, "main:de-de": 0.2, \
                                        "main:fr-fr: 0.3, "main:zh-zh": 0.3 }',
            default=None,
            type=lambda uf: utils.eval_str_dict(uf, type=str),
        )
        parser.add_argument(
            "--only-target-token-percent",
            help='Only generating the first N (N from 0.00 to 1.00) of tokens in target language'
                    'Only applied in on-the-fly BT.'
                    'Set 1 to disable.',
            default=1,
            type=float,
        )
        parser.add_argument(
            "--only-target-token-before-step",
            help='Only generating tokens in target language for the first N steps of on-the-fly BT.'
                'Set 0 to disable.',
            default=0,
            type=int,
        )
        MultilingualDatasetManager.add_args(parser)
        # fmt: on   

    @classmethod
    def setup_data_manager(cls, args, lang_pairs, eval_lang_pairs, langs, dicts, sampling_method, training):
        return MultilingualSemisupervisedDatasetManager(
            args, lang_pairs, eval_lang_pairs, langs, dicts, sampling_method, training
        )

    def get_data_paths_and_lang_pairs(self, split):
        """Get the data paths and lang pairs to be loaded.

        Args:
            split (str): data subset to be loaded (e.g. train, valid, test)

        for 'valid' or 'test', this will return:
            {'main': '/.../data-bin'}, {'main': ['en-de', 'de-zh', 'de-en', 'zh-de']}

        for 'train', this will return:
            {'main': '/.../data-bin'}, {'main': ['en-None', 'zh-None', 'de-None', 'en-de', 'de-zh', 'de-en', 'zh-de']}
        """
        datapaths = {"main": self.args.data}
        if split == getattr(self.args, "train_subset", None):
            # monolingual data + parallel data for training set
            lang_pairs = {"main": [f"{lang}-None" for lang in self.training_mono_langs] + self.lang_pairs} 
        else:
            lang_pairs = {"main": self.eval_lang_pairs}
        return datapaths, lang_pairs

    def get_split_num_data_shards(self, split):
        """Get the dictionary mapping language pair 
        to the number of available shards.

        for 'valid' or 'test', this will return:
            {'main:en-de': 1, 'main:de-zh': 1, 'main:de-en': 1, 'main:zh-de': 1}

        for 'train', this will return:
            {'main:zh-None': 1, 'main:en-None': 1, 'main:de-None': 1}

        Args:
            split (str): data subset to be loaded (e.g. train, valid, test)

        Returns:
            dict: the dictionary mapping language pair to the number of available shards.
        """
        if split in self._num_shards_dict:
            return self._num_shards_dict[split]
        num_shards_dict = {}
        data_paths, lang_pairs = self.get_data_paths_and_lang_pairs(split)

        for data_category, paths in data_paths.items():
            if data_category not in lang_pairs:
                continue
            paths = utils.split_paths(paths)
            shards_dict = self._get_shard_num_dict(split, paths)
            lang_dirs = [
                lang_pair.split("-") for lang_pair in lang_pairs[data_category]
            ]
            lang_dirs = [x if len(x) > 1 else (x[0], x[0]) for x in lang_dirs]
            for src, tgt in lang_dirs:
                key = self.get_dataset_key(data_category, src, tgt)
                if src != tgt:
                    if f"{src}-{tgt}" in shards_dict:
                        num_shards_dict[key] = shards_dict[f"{src}-{tgt}"]
                    elif f"{tgt}-{src}" in shards_dict:
                        # follow the fairseq tradition to use reversed direction data if it is not available
                        num_shards_dict[key] = shards_dict[f"{tgt}-{src}"]
                else:
                    if f"{src}-None" in shards_dict:
                        num_shards_dict[key] = shards_dict[f"{src}-None"]

        self._num_shards_dict[split] = num_shards_dict
        logger.info(f"[{split}] num of shards: {num_shards_dict}")
        return num_shards_dict

    def get_split_data_param_list(self, split, epoch, shard_epoch=None):
        """Get the data parameter of the dataset for each language pair.
        
        Return a list of dictionaries. Each dictionary contains the data 
        parameter for a lang pair. For parallel language pairs the dictionary 
        will be like:
        {   
            'key': 'main:en-de', 
            'data_path': '/.../data-bin', 
            'split': 'valid', 
            'src': 'en', 
            'src_dict': <fairseq.data.dictionary.Dictionary object>, 
            'tgt': 'de', 
            'tgt_dict': <fairseq.data.dictionary.Dictionary object>, 
            'data_category': 'main', 
            'langtok_spec': ('src', 'tgt')
        }

        For mono language, we leave the tgt field as None:
        {
            'key': 'main:en-None', 
            'data_path': '/.../data-bin', 
            'split': 'train', 
            'src': 'en', 
            'src_dict': <fairseq.data.dictionary.Dictionary object>, 
            'tgt': None, 
            'tgt_dict': None, 
            'data_category': 'main', 
            'langtok_spec': ('src', 'tgt')
        }

        Args:
            split (str): data subset to be loaded (e.g. train, valid, test)
            epoch (int): get data parameters for this epoch. ignored if *shard_epoch* != None 
            shard_epoch (int, optional): When given, replace *epoch*. Defaults to None.

        Returns:
            List[Dict]: A list containing dictionaries of data parameters
        """
        param_list = []
        data_paths, lang_pairs = self.get_data_paths_and_lang_pairs(split)
        logger.info(f"langtoks settings: {self.args.langtoks}")
        split_num_shards_dict = self.get_split_num_data_shards(split)
        for data_category, paths in data_paths.items():
            if data_category not in lang_pairs:
                continue
            paths = utils.split_paths(paths)
            assert len(paths) > 0
            if len(paths) > 1:
                self._has_sharded_data = True
            if split != getattr(self.args, "train_subset", None):
                # if not training data set, use the first shard for valid and test
                paths = paths[:1]

            if data_category in self.args.langtoks:
                lang_tok_spec = self.args.langtoks[data_category]
            else:
                # default to None
                lang_tok_spec = (None, None)

            # infer langcode
            lang_dirs = [
                lang_pair.split("-") for lang_pair in lang_pairs[data_category]
            ]
            lang_dirs = [x if len(x) > 1 else (x[0], x[0]) for x in lang_dirs]
            for src, tgt in lang_dirs:
                assert src is not None or data_category == "mono_dae", (
                    f"error: src={src}, " "tgt={tgt} for data_category={data_category}"
                )
                # logger.info(f"preparing param for {data_category}: {src} - {tgt}")
                key = self.get_dataset_key(data_category, src, tgt)
                data_path = self.get_split_data_path(
                    paths, epoch, shard_epoch, split_num_shards_dict[key]
                )

                param_list.append(
                    {
                        "key": key,
                        "data_path": data_path,
                        "split": split,
                        "src": src,
                        "src_dict": self.get_source_dictionary(src)
                        if src and data_category != "mono_dae"
                        else None,

                        # we leave the tgt field as None for mono datasets, 
                        "tgt": tgt if tgt != "None" else None,
                        "tgt_dict": self.get_target_dictionary(tgt) if tgt != "None" else None,

                        "data_category": data_category,
                        "langtok_spec": lang_tok_spec,
                    }
                )
        return param_list

    def load_split_mono_datasets(self, split, training, epoch=1, combine=False, shard_epoch=None, **kwargs):
        """Load multiple monolingual datasets

        Args:
            split (str): data subset to be loaded (e.g. train)
            training (_type_): not used
            epoch (int, optional): get data parameters for this epoch. ignored if *shard_epoch* != None. Defaults to 1.
            combine (bool, optional): whether to combine all the shards. If False, only load the first shard. Defaults to False.
            shard_epoch (int, optional): When given, replace *epoch*. Defaults to None.

        Returns:
            Tuple: a list of monolingual datasets and corresponding data parameters
        """
        data_param_list = self.get_split_data_param_list(
            split, epoch, shard_epoch=shard_epoch
        )
        data_param_list = [param for param in data_param_list if param["tgt"] == None]
        datasets = [
            (
                param["key"],
                self.load_a_mono_dataset(
                    combine=combine,
                    **param,
                ),
            )
            for param in data_param_list
        ]
        return datasets, data_param_list

    def load_a_mono_dataset(
        self,
        split,
        data_path,
        src,
        src_dict,
        combine,
        **extra_kwargs
    ):
        """Load a raw monolingual dataset

        Args:
            split (str): data subset to be loaded (e.g. train)
            data_path (str): path to load data
            src (str): mono language
            src_dict (fairseq.data.dictionary.Dictionary): dictionary of mono language
            combine (bool): whether to combine all the shards. If False, only load the first shard

        Raises:
            FileNotFoundError: Raise error if data file not found

        Returns:
            _type_: a raw monolingual dataset
        """
        dataset_impl = self.args.dataset_impl
        upsample_primary = self.args.upsample_primary

        src_datasets = []
        tgt = "None"
        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else "")

            # infer langcode
            if self.split_exists(split_k, src, tgt, src, data_path, dataset_impl):
                prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
            else:
                if k > 0:
                    break
                else:
                    logger.error(
                        f"Dataset not found: {data_path}, {split_k}, {src}, {tgt}"
                    )
                    raise FileNotFoundError(
                        "Dataset not found: {} ({})".format(split, data_path)
                    )

            src_dataset = self.load_data(prefix + src, src_dict, dataset_impl)
            src_datasets.append(src_dataset)

            logger.info(
                "{} {} {}-{} {} examples".format(
                    data_path, split_k, src, tgt, len(src_datasets[-1])
                )
            )

            if not combine:
                break

        if len(src_datasets) == 1:
            src_dataset = src_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)

        return src_dataset

    def load_split_para_datasets(self, split, training, epoch=1, combine=False, shard_epoch=None, **kwargs):
        """Load multiple para datasets

        Args:
            split (str): data subset to be loaded (e.g. train)
            training (_type_): not used
            epoch (int, optional): get data parameters for this epoch. ignored if *shard_epoch* != None. Defaults to 1.
            combine (bool, optional): whether to combine all the shards. If False, only load the first shard. Defaults to False.
            shard_epoch (int, optional): When given, replace *epoch*. Defaults to None.

        Returns:
            Tuple: a list of para datasets and corresponding data parameters
        """
        data_param_list = self.get_split_data_param_list(
            split, epoch, shard_epoch=shard_epoch
        )
        data_param_list = [param for param in data_param_list if param["tgt"] != None]
        langpairs_sharing_datasets = (
            {} if self.args.enable_reservsed_directions_shared_datasets else None
        )
        datasets = [
            (
                param["key"],
                self.load_a_dataset(
                    combine=combine,
                    langpairs_sharing_datasets=langpairs_sharing_datasets,
                    **param,
                ),
            )
            for param in data_param_list
        ]
        return datasets, data_param_list

    def construct_multi_mt_dataset(self, split, para_datasets, para_data_param_list, epoch, shard_epoch, proportion):
        """Construct a dataset containing all the parallel datasets

        Args:
            split (str): the split of the data, e.g. 'train', 'valid' or 'test'.
            para_datasets (list): list of para datasets
            para_data_param_list (_type_): list of data parameters for *para_datasets*
            epoch (int): starting epoch number.
            shard_epoch (int): the real epoch number for shard selection.
            proportion (float): Proportion of mt data to virtual_epoch_size
            

        Returns:
            _type_: _description_
        """
        mt_datasets = []
        mt_data_param_list = []
        for lang_pair in self.lang_pairs:
            src_lang, tgt_lang = lang_pair.split("-")
            dataset_key = self.get_dataset_key("main", src_lang, tgt_lang)
            para_data_param = [i for i in para_data_param_list if i['key'] == dataset_key][0]

            mt_datasets.append((lang_pair, para_datasets[dataset_key]))
            mt_data_param_list.append(para_data_param)

        sample_ratios = self.get_sampling_ratios(mt_data_param_list, mt_datasets, epoch, task=MT_NAME)
        return SampledMultiEpochDataset(
            OrderedDict(mt_datasets),
            epoch=epoch,
            shard_epoch=shard_epoch,
            sampling_ratios=sample_ratios,
            eval_key=None,
            collate_format=CollateFormat.ordered_dict,
            virtual_size=self.args.virtual_data_size,
            split=split,
            virtual_epoch_size=int(self.args.virtual_epoch_size * proportion),
            shared_collater=True,
        )

    def get_sampling_ratios(self, data_param_list, datasets, epoch, shard_epoch=None, task=None):
        """Get sampling ratio for each dataset

        Args:
            data_param_list (List): list of data parameters for each dataset
            datasets (List): list of datasets
            epoch (int): starting epoch number.
            shard_epoch (int): the real epoch number for shard selection.
            task (str, optional): 'bt' or 'dae' or 'mt'. Defaults to None.

        Returns:
            List: list of sampling ratios for each dataset
        """
        assert task in [BT_NAME, DAE_NAME, MT_NAME], f"task=={task} should be in {[BT_NAME, DAE_NAME, MT_NAME]}."
        if task == BT_NAME:
            sampling_weights_from_file = self.args.bt_sampling_weights_from_file 
            sampling_weights = self.args.bt_sampling_weights 
        elif task == DAE_NAME:
            sampling_weights_from_file = self.args.dae_sampling_weights_from_file 
            sampling_weights = self.args.dae_sampling_weights 
        elif task == MT_NAME:
            sampling_weights_from_file = self.args.sampling_weights_from_file 
            sampling_weights = self.args.sampling_weights 

        if sampling_weights_from_file:
            weights = load_sampling_weights(self.args.sampling_weights_from_file)
            sample_ratios = [weights[k] for k, _ in datasets]
            logger.info(
                f"| [{task}] ignoring --sampling-weights when loading sampling weights "
                f"from file {sampling_weights_from_file}"
            )
        elif sampling_weights:
            sample_ratios = [sampling_weights[k] for k, _ in datasets]
        else:
            sample_ratios = self.get_train_sampling_ratios(
                data_param_list, datasets, epoch, shard_epoch, task
            )

        if sample_ratios is not None:
            logger.info(
                f"| [{task}] Upsample ratios: "
                "{}".format(
                    list(zip(map(lambda x: x["key"], data_param_list), sample_ratios))
                )
            )
            assert len(sample_ratios) == len(datasets)
        return sample_ratios

    def construct_multi_bt_dataset(self, split, mono_datasets, mono_data_param_list, epoch, shard_epoch, proportion):
        """Construct a dataset containing all the bt datasets

        Args:
            split (str): the split of the data, e.g. 'train', 'valid' or 'test'.
            mono_datasets (list): list of mono datasets
            mono_data_param_list (_type_): list of data parameters for *mono_datasets*
            epoch (int): starting epoch number.
            shard_epoch (int): the real epoch number for shard selection.
            proportion (float): Proportion of BT data to virtual_epoch_size

        Returns:
            _type_: _description_
        """
        bt_datasets = []
        bt_data_param_list = []
        for lang_pair in self.lang_pairs:
            other_lang, original_lang = lang_pair.split("-")

            dataset_key = self.get_dataset_key("main", original_lang, "None")
            mono_data_param = [i for i in mono_data_param_list if i['key'] == dataset_key][0]
            mono_data_param["lang_pair"] = lang_pair
            bt_data_param_list.append(mono_data_param)

            _, ori_langtok_spec = mono_data_param["langtok_spec"]
            oth_dict = self.get_source_dictionary(other_lang)
            ori_dict = mono_data_param["src_dict"]

            ori_dataset = mono_datasets[dataset_key]
            ori_dataset = self.tgt_dataset_tranform_func(
                source_lang=other_lang, 
                target_lang=original_lang, 
                dataset=ori_dataset, 
                spec=ori_langtok_spec
            )
            ori_dataset= LanguagePairDataset(
                ori_dataset,
                ori_dataset.sizes,
                ori_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                src_lang_id=_lang_id(self.lang_dict, other_lang)
                if self.args.enable_lang_ids and self.lang_dict is not None
                else None,
                tgt_lang_id=_lang_id(self.lang_dict, original_lang)
                if self.args.enable_lang_ids and self.lang_dict is not None
                else None,
            )
            
            bt_dataset = WrapperBacktranslationDataset(
                tgt_dataset=ori_dataset,
                src_dict=oth_dict,
                tgt_dict=ori_dict,
                backtranslation_fn=None,
                cuda=(not self.args.cpu)
            )

            bt_datasets.append((lang_pair, bt_dataset))

        sample_ratios = self.get_sampling_ratios(bt_data_param_list, bt_datasets, epoch, task=BT_NAME)
        return SampledMultiEpochDataset(
            OrderedDict(bt_datasets),
            epoch=epoch,
            shard_epoch=shard_epoch,
            sampling_ratios=sample_ratios,
            eval_key=None,
            collate_format=CollateFormat.ordered_dict,
            virtual_size=self.args.virtual_data_size,
            split=split,
            virtual_epoch_size=int(self.args.virtual_epoch_size * proportion),
            shared_collater=False
        )

    def construct_multi_dae_dataset(self, split, mono_datasets, mono_data_param_list, epoch, shard_epoch, proportion):
        """Construct a dataset containing all the dae datasets

        Args:
            split (str): the split of the data, e.g. 'train', 'valid' or 'test'.
            mono_datasets (list): list of mono datasets
            mono_data_param_list (_type_): list of data parameters for *mono_datasets*
            epoch (int): starting epoch number.
            shard_epoch (int): the real epoch number for shard selection.
            proportion (float): Proportion of DAE data to virtual_epoch_size

        Returns:
            _type_: _description_
        """
        dae_datasets = []
        dae_data_param_list = []
        for lang in self.training_mono_langs:
            dataset_key = self.get_dataset_key("main", lang, "None")
            mono_data_param = [i for i in mono_data_param_list if i['key'] == dataset_key][0]
            mono_data_param["lang_pair"] = f"{lang}-{lang}"
            dae_data_param_list.append(mono_data_param)

            _, clean_langtok_spec = mono_data_param["langtok_spec"]
            noised_dict = mono_data_param["src_dict"]
            clean_dict = self.get_target_dictionary(lang)

            clean_dataset = mono_datasets[dataset_key]
            clean_dataset = self.tgt_dataset_tranform_func(
                source_lang=lang,
                target_lang=lang,
                dataset=clean_dataset,
                spec=clean_langtok_spec
            )
            clean_dataset = LanguagePairDataset(
                clean_dataset,
                clean_dataset.sizes,
                noised_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                src_lang_id=_lang_id(self.lang_dict, lang)
                if self.args.enable_lang_ids and self.lang_dict is not None
                else None,
                tgt_lang_id=_lang_id(self.lang_dict, lang)
                if self.args.enable_lang_ids and self.lang_dict is not None
                else None
            )

            dae_dataset = DenoisingDataset(
                max_word_shuffle_distance=self.args.max_word_shuffle_distance,
                word_dropout_prob=self.args.word_dropout_prob,
                word_blanking_prob=self.args.word_blanking_prob,
                tgt_dataset=clean_dataset,
                src_dict=noised_dict,
                tgt_dict=clean_dict,
                lang_tok_id=self.get_decoder_langtok(lang, clean_langtok_spec),
                cuda=False,
                seed=self.seed,
            )

            dae_datasets.append((f"{lang}-{lang}", dae_dataset))

        sample_ratios = self.get_sampling_ratios(dae_data_param_list, dae_datasets, epoch, task=DAE_NAME)
        return SampledMultiEpochDataset(
            OrderedDict(dae_datasets),
            epoch=epoch,
            shard_epoch=shard_epoch,
            sampling_ratios=sample_ratios,
            eval_key=None,
            collate_format=CollateFormat.ordered_dict,
            virtual_size=self.args.virtual_data_size,
            split=split,
            virtual_epoch_size=int(self.args.virtual_epoch_size*proportion),
            shared_collater=True,
        )

    def load_task_dataset(
        self, split, task, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        if task == BT_NAME:
            return self.load_bt_dataset(split, epoch, combine, shard_epoch, **kwargs)
        elif task == DAE_NAME:
            return self.load_dae_dataset(split, epoch, combine, shard_epoch, **kwargs)
        elif task == MT_NAME:
            return self.load_mt_dataset(split, epoch, combine, shard_epoch, **kwargs)

    def load_bt_dataset(
        self, split, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        """Load dataset for *split* (bt)

        Args:
            split (str): data subset to be loaded (e.g. train, test, valid)
            epoch (int, optional): starting epoch number. Defaults to 0.
            combine (bool, optional): whether to combine all the shards. If False, only load the first shard. Defaults to False.
            shard_epoch (int, optional): the real epoch number for shard selection.. Defaults to None.

        Returns:
            _type_: loaded dataset
        """
        # Load all single monolingual datasets
        mono_datasets, mono_data_param_list = self.load_split_mono_datasets(
            split, True, epoch, combine, shard_epoch=shard_epoch, **kwargs
        )
        mono_datasets = OrderedDict(mono_datasets)

        # construct back-translation dataset
        multi_bt_dataset = self.construct_multi_bt_dataset(split, mono_datasets, mono_data_param_list, epoch, shard_epoch, 1.0/3.0)

        return multi_bt_dataset

    def load_dae_dataset(
        self, split, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        """Load dataset for *split* (dae)

        Args:
            split (str): data subset to be loaded (e.g. train, test, valid)
            epoch (int, optional): starting epoch number. Defaults to 0.
            combine (bool, optional): whether to combine all the shards. If False, only load the first shard. Defaults to False.
            shard_epoch (int, optional): the real epoch number for shard selection.. Defaults to None.

        Returns:
            _type_: loaded dataset
        """
        # Load all single monolingual datasets
        mono_datasets, mono_data_param_list = self.load_split_mono_datasets(
            split, True, epoch, combine, shard_epoch=shard_epoch, **kwargs
        )
        mono_datasets = OrderedDict(mono_datasets)

        # construct denoising auto-encoding dataset
        multi_dae_dataset = self.construct_multi_dae_dataset(split, mono_datasets, mono_data_param_list, epoch, shard_epoch, 1.0/3.0)

        return multi_dae_dataset

    def load_mt_dataset(
        self, split, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        """Load dataset for *split* (mt)

        Args:
            split (str): data subset to be loaded (e.g. train, test, valid)
            epoch (int, optional): starting epoch number. Defaults to 0.
            combine (bool, optional): whether to combine all the shards. If False, only load the first shard. Defaults to False.
            shard_epoch (int, optional): the real epoch number for shard selection.. Defaults to None.

        Returns:
            _type_: loaded dataset
        """
        # Load all para datasets
        para_datasets, para_data_param_list = self.load_split_para_datasets(
            split, True, epoch, combine, shard_epoch=shard_epoch, **kwargs
        )
        para_datasets = OrderedDict(para_datasets)

        # construct mt dataset
        multi_mt_dataset = self.construct_multi_mt_dataset(split, para_datasets, para_data_param_list, epoch, shard_epoch, 1.0/3.0)
        return multi_mt_dataset

    def load_task_dataset(self, task, split, epoch=0, combine=False, shard_epoch=None, **kwargs):
        if task == BT_NAME:
            return self.load_bt_dataset(split, epoch, combine, shard_epoch, **kwargs)
        elif task == DAE_NAME:
            return self.load_dae_dataset(split, epoch, combine, shard_epoch, **kwargs)
        elif task == MT_NAME:
            return self.load_mt_dataset(split, epoch, combine, shard_epoch, **kwargs)

    def load_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        """Load dataset for *split*

        Args:
            split (str): data subset to be loaded (e.g. train, test, valid)
            training (bool): if training or not
            epoch (int, optional): starting epoch number. Defaults to 0.
            combine (bool, optional): whether to combine all the shards. If False, only load the first shard. Defaults to False.
            shard_epoch (int, optional): the real epoch number for shard selection.. Defaults to None.

        Returns:
            _type_: loaded dataset
        """
        if training and split == getattr(self.args, "train_subset", None):
            # zip all the train-tasks as a round-robin dataset
            rr_zip_dataset = WrapperRoundRobinZipDatasets(
                {
                    task_name:  self.load_task_dataset(task_name, split, epoch, combine, shard_epoch, **kwargs)
                        for task_name in self.args.train_tasks
                }
            )
            rr_zip_dataset.ordered_indices()
            return rr_zip_dataset

        else:
            datasets, data_param_list = self.load_split_datasets(
                split, training, epoch, combine, shard_epoch=shard_epoch, **kwargs
            )
            return self.load_into_concat_dataset(split, datasets, data_param_list)

    def get_train_dataset_sizes(
        self, data_param_list, datasets, epoch, shard_epoch=None, task=None
    ):
        assert task in [BT_NAME, DAE_NAME, MT_NAME], f"task=={task} should be in {[BT_NAME, DAE_NAME, MT_NAME]}."
        num_shards = [
            self.get_split_num_data_shards(param["split"])[param["key"]]
            for param in data_param_list
        ]
        data_sizes = []
        for (key, d), num_shard in zip(datasets, num_shards):
            my_data_sizes = self._training_data_sizes[task][key]
            shard_ind = self.get_shard_id(num_shard, epoch, shard_epoch)
            if shard_ind not in my_data_sizes:
                my_data_sizes[shard_ind] = len(d)
            known_size = max(my_data_sizes.values())
            data_sizes.append(
                # If we don't know the data size of the shard yet,
                # use the the max known data size to approximate.
                # Note that we preprocess shards by a designated shard size
                # and put any remaining data at the end into the last shard so
                # the max shard size approximation is almost correct before loading
                # the last shard; after loading the last shard, it will have the
                # exact data sizes of the whole data size.
                (key, sum(my_data_sizes.get(i, known_size) for i in range(num_shard)))
            )
        logger.info(
            f"estimated total data sizes of all shards used in sampling ratios: {data_sizes}. "
            "Note that if the data a shard has not been loaded yet, use the max known data size to approximate"
        )
        return [s for _, s in data_sizes]

    def get_train_sampling_ratios(
        self, data_param_list, datasets, epoch=1, shard_epoch=None, task=None
    ):
        data_sizes = self.get_train_dataset_sizes(
            data_param_list, datasets, epoch, shard_epoch, task
        )
        sampling_func = self.sampling_method.sampling_method_selector()
        sample_ratios = sampling_func(data_sizes) if sampling_func is not None else None
        return sample_ratios

    def update_step(self, num_updates):
        if (not hasattr(self, "if_mask")) or (not self.if_mask):
            return
        if self.only_target_token_percent != 1 and self.only_target_token_before_step != 0:
            if num_updates + 1 >= self.only_target_token_before_step:
                for lang_pair in self.lang_pairs:
                    tgt_lang = lang_pair.split('-')[-1].strip()
                    self.tgt_lang_mask_words_idx[tgt_lang] = self.tgt_lang_mask_lang_tokens_idx[tgt_lang]
                self.if_mask = False
                logger.info(
                    "Disable masked tokens in on-the-fly BT."
                )