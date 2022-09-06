import logging
import torch
import math
import json
import numpy as np
from argparse import Namespace
from fairseq import metrics, utils
from fairseq.data import data_utils, encoders
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.data.multilingual.multilingual_semisupervised_data_manager import (
    BT_NAME,
    DAE_NAME,
    MT_NAME, 
    MultilingualSemisupervisedDatasetManager
)
from fairseq.search import LengthConstrainedBeamSearch
from fairseq.tasks import register_task
from fairseq.tasks.semisupervised_translation import parse_lambda_config
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.sequence_generator import TokenMaskedSequenceGenerator
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.utils import csv_str_list, FileContentsAction

logger = logging.getLogger(__name__)
EVAL_BLEU_ORDER = 4

@register_task("multilingual_semisupervised_translation")
class MultilingualSemisupervisedTranslationTask(TranslationMultiSimpleEpochTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off

        # langauge settings
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs: en-de,en-de,de-fr,fr-de',
                            action=FileContentsAction)

        # inference setting (only valid when eval or generate)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')
        
        # training settings
        parser.add_argument("--train-tasks", default="bt,dae,mt", type=csv_str_list,
            help="comma separated task, select from bt,dae,mt"
        )

        # denoising auto-encoding settings
        parser.add_argument('--lambda-denoising-config', default="0:1,100000:0.1,300000:0", type=str, metavar='CONFIG',
                            help='Cross-entropy reconstruction coefficient (denoising autoencoding)'
                                'use fixed weight during training if set to floating point number. '
                                'use piecewise linear function over number of updates to schedule the '
                                'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--max-word-shuffle-distance', default=3, type=int, metavar='N',
                            help='maximum word shuffle distance for denoising autoencoding data generation')
        parser.add_argument('--word-dropout-prob', default=0.1, type=float, metavar='N',
                            help='word dropout probability for denoising autoencoding data generation')
        parser.add_argument('--word-blanking-prob', default=0.1, type=float, metavar='N',
                            help='word blanking probability for denoising autoencoding data generation')

        # back-translation settings
        parser.add_argument('--lambda-otf-bt-config', default="1.0", type=str, metavar='CONFIG',
                            help='cross-entropy reconstruction coefficient (on-the-fly back-translation parallel data)'
                                'use fixed weight during training if set to floating point number. '
                                'use piecewise linear function over number of updates to schedule the '
                                'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--bt-beam-size', type=int, default=1)
        parser.add_argument('--bt-min-len-a', type=float, default=0)
        parser.add_argument('--bt-min-len-b', type=float, default=1)
        parser.add_argument('--bt-max-len-a', type=float, default=1.3)
        parser.add_argument('--bt-max-len-b', type=float, default=5)

        # debugging settings
        parser.add_argument('--print-train-samples', default=False, action='store_true',
                            help="print training samples in each step")
        parser.add_argument('--bpe-symbol', default="sentencepiece", type=str,
                            help='post process bpe symbol'
                                'apply when --print-train-samples is True')

        # evaluation settings
        parser.add_argument("--eval-lang-pairs", default=None, type=csv_str_list,
            help="comma separated language pairs which used to evaluate the model"
        )
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                'required if using --eval-bleu; use "space" to '
                                'disable detokenization; see fairseq.data.encoders '
                                'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')

        SamplingMethod.add_arguments(parser)
        MultilingualSemisupervisedDatasetManager.add_args(parser)
        # fmt: on

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        if training:
            self.eval_lang_pairs = args.eval_lang_pairs
        else:
            self.eval_lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]

        
        self.data_manager = MultilingualSemisupervisedDatasetManager.setup_data_manager(
            args, self.lang_pairs, self.eval_lang_pairs, langs, dicts, self.sampling_method, training
        )
        self.lang_pairs = self.data_manager.lang_pairs
        self.lambda_otf_bt, self.lambda_otf_bt_steps = parse_lambda_config(
            args.lambda_otf_bt_config
        )
        self.lambda_denoising, self.lambda_denoising_steps = parse_lambda_config(
            args.lambda_denoising_config
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = MultilingualSemisupervisedDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )
        return cls(args, langs, dicts, training)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        """
        need_def_bt_func = False

        if self.training and split == getattr(self.args, "train_subset", None) and split in self.datasets:
            for task in self.args.train_tasks:
                if task in self.datasets[split].datasets:
                    dataset = self.datasets[split].datasets[task]
                    if self.has_sharded_data(split):
                        if self.args.virtual_epoch_size is not None:
                            if dataset.load_next_shard:
                                shard_epoch = dataset.shard_epoch
                            else:
                                # no need to load next shard so skip loading
                                # also this avoid always loading from beginning of the data
                                continue
                        else:
                            shard_epoch = epoch
                else:
                    # estimate the shard epoch from virtual data size and virtual epoch size
                    shard_epoch = self.data_manager.estimate_global_pass_epoch(epoch)

                if task == BT_NAME:
                    need_def_bt_func = True

                logger.info(f"loading data for {split} {task} epoch={epoch}/{shard_epoch}")
                logger.info(f"mem usage: {data_utils.get_mem_usage()}")
                logger.info(f"old {split} {task} dataset deleted manually")
                del self.datasets[split].datasets[task]
                
                self.datasets[split].datasets[task] = self.data_manager.load_task_dataset(
                    split,
                    task,
                    epoch=epoch,
                    combine=combine,
                    shard_epoch=shard_epoch,
                    **kwargs,
                )
            self.datasets[split].refresh()

        else:
            if self.training and split == getattr(self.args, "train_subset", None) and BT_NAME in self.args.train_tasks:
                need_def_bt_func = True

            # estimate the shard epoch from virtual data size and virtual epoch size
            shard_epoch = self.data_manager.estimate_global_pass_epoch(epoch)
            self.datasets[split] = self.data_manager.load_dataset(
                split,
                self.training,
                epoch=epoch,
                combine=combine,
                shard_epoch=shard_epoch,
                **kwargs,
            )


        if need_def_bt_func:
            for lang_pair in self.lang_pairs:
                src_lang, _ = lang_pair.split("-")

                def backtranslate_fn(sample, tgt_lang=src_lang):
                    with torch.no_grad():
                        _, tgt_langtok_spec = self.args.langtoks["main"]
                        if tgt_langtok_spec:
                            tgt_lang_tok = self.data_manager.get_decoder_langtok(
                                tgt_lang, tgt_langtok_spec
                            )
                            src_tokens = sample["net_input"]["src_tokens"]
                            bsz = src_tokens.size(0)
                            prefix_tokens = (
                                torch.LongTensor([[tgt_lang_tok]]).expand(bsz, 1).to(src_tokens)
                            )
                        return self.bt_generator.generate(
                            models=None,
                            sample=sample,
                            prefix_tokens=prefix_tokens,
                            masked_tokens=self.data_manager.tgt_lang_mask_words_idx[tgt_lang] 
                                if hasattr(self.data_manager, "tgt_lang_mask_words_idx") 
                                else None,
                        )

                self.datasets[split].datasets[BT_NAME].datasets[
                    self.datasets[split].datasets[BT_NAME].keys.index(lang_pair)
                ].set_backtranslation_fn(backtranslate_fn)

    def build_bt_generator(
        self,
        models,
        beam_size,
        min_len_a,
        min_len_b,
        max_len_a,
        max_len_b,
        max_len
    ):
        """Build a generator for back-translation

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int): beam width
            max_len_a/b (int): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int): the maximum length of the generated output
                (not including end-of-sentence)
        Returns:
            ~fairseq.sequence_generator.TokenMaskedSequenceGenerator: generator used to back-translate
        """
        
        return TokenMaskedSequenceGenerator(
            models=models,
            tgt_dict=self.target_dictionary,
            search_strategy=LengthConstrainedBeamSearch(
                tgt_dict=self.target_dictionary, 
                min_len_a=min_len_a, 
                min_len_b=min_len_b, 
                max_len_a=max_len_a, 
                max_len_b=max_len_b
            ),
            beam_size=beam_size,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            max_len=max_len
        )

    def build_model(self, args, from_checkpoint=False):
        """Build a model.
        
        Most of it is done by the parent class. 
        We need to prepare a sequence generator for back-translation 
        here, which will be used when loading data. 
        """
        model = super().build_model(args, from_checkpoint)

        self.bt_generator = self.build_bt_generator(
            [model], 
            beam_size=args.bt_beam_size,
            min_len_a=args.bt_min_len_a,
            min_len_b=args.bt_min_len_b,
            max_len_a=args.bt_max_len_a,
            max_len_b=args.bt_max_len_b,
            max_len=args.max_source_positions,
        )

        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            extra_gen_cls_kwargs=None
            if not getattr(args, "keep_inference_langtok", False):
                for tgt_lang in self.target_langs:
                    _, tgt_langtok_spec = self.args.langtoks["main"]
                    if tgt_langtok_spec:
                        tgt_lang_tok = self.data_manager.get_decoder_langtok(
                            tgt_lang, tgt_langtok_spec
                        )
                        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
                        if "symbols_to_strip_from_output" not in extra_gen_cls_kwargs:
                            extra_gen_cls_kwargs["symbols_to_strip_from_output"]=set()
                        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = (
                            extra_gen_cls_kwargs["symbols_to_strip_from_output"].union({tgt_lang_tok})
                        )
                    
            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.bleu_sequence_generator = super(
                    TranslationMultiSimpleEpochTask, 
                    self
                ).build_generator(
                    [model], 
                    Namespace(**gen_args),
                    extra_gen_cls_kwargs=extra_gen_cls_kwargs
            )

        return model

    def reduce_metrics(self, logging_outputs, criterion):
        with metrics.aggregate():
            super().reduce_metrics(logging_outputs, criterion)

            # reduce loss
            reduce_loss_keys = [f"{DAE_NAME}_loss", f"{DAE_NAME}_nll_loss", f"{BT_NAME}_loss", f"{BT_NAME}_nll_loss", f"{MT_NAME}_loss", f"{MT_NAME}_nll_loss"]
            for k in reduce_loss_keys:
                sample_size = sum(log.get(k.split("_")[0]+"_sample_size", 0) for log in logging_outputs)
                ntokens = sum(log.get(k.split("_")[0]+"_ntokens", 0) for log in logging_outputs)
                k_sum = sum(log.get(k, 0) for log in logging_outputs)
                if sample_size != 0.0 and ntokens != 0.0:
                    if "nll_loss" in k:
                        metrics.log_scalar(k, k_sum / ntokens / math.log(2), ntokens, priority=5, round=3)
                    else :
                        metrics.log_scalar(k, k_sum / sample_size / math.log(2), sample_size, priority=5, round=3)
            
            # reduce stats
            for k in ["sample_size", "nsentences", "ntokens"]:
                metrics.log_scalar(k, sum(l[k] for l in logging_outputs))
            
            if self.args.eval_bleu:

                def sum_logs(key):
                    import torch

                    result = sum(log.get(key, 0) for log in logging_outputs)
                    if torch.is_tensor(result):
                        result = result.cpu()
                    return result

                counts, totals = [], []
                for i in range(EVAL_BLEU_ORDER):
                    counts.append(sum_logs("_bleu_counts_" + str(i)))
                    totals.append(sum_logs("_bleu_totals_" + str(i)))

                if max(totals) > 0:
                    # log counts as numpy arrays -- log_scalar will sum them correctly
                    metrics.log_scalar("_bleu_counts", np.array(counts))
                    metrics.log_scalar("_bleu_totals", np.array(totals))
                    metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                    metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                    def compute_bleu(meters):
                        import inspect

                        try:
                            from sacrebleu.metrics import BLEU

                            comp_bleu = BLEU.compute_bleu
                        except ImportError:
                            # compatibility API for sacrebleu 1.x
                            import sacrebleu

                            comp_bleu = sacrebleu.compute_bleu

                        fn_sig = inspect.getfullargspec(comp_bleu)[0]
                        if "smooth_method" in fn_sig:
                            smooth = {"smooth_method": "exp"}
                        else:
                            smooth = {"smooth": "exp"}
                        bleu = comp_bleu(
                            correct=meters["_bleu_counts"].sum,
                            total=meters["_bleu_totals"].sum,
                            sys_len=meters["_bleu_sys_len"].sum,
                            ref_len=meters["_bleu_ref_len"].sum,
                            **smooth,
                        )
                        return round(bleu.score, 2)

                    metrics.log_derived("bleu", compute_bleu)

    def update_step(self, num_updates):
        # update loss weight
        def lambda_step_func(config, n_iter):
            """
            Update a lambda value according to its schedule configuration.
            """
            ranges = [
                i
                for i in range(len(config) - 1)
                if config[i][0] <= n_iter < config[i + 1][0]
            ]
            if len(ranges) == 0:
                assert n_iter >= config[-1][0]
                return config[-1][1]
            assert len(ranges) == 1
            i = ranges[0]
            x_a, y_a = config[i]
            x_b, y_b = config[i + 1]
            return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)

        if self.lambda_denoising_steps is not None:
            self.lambda_denoising = lambda_step_func(
                self.lambda_denoising_steps, num_updates
            )
            metrics.log_scalar("dae_lam", self.lambda_denoising, weight=0, round=4)
        if self.lambda_otf_bt_steps is not None:
            self.lambda_otf_bt = lambda_step_func(
                self.lambda_otf_bt_steps, num_updates
            )
            metrics.log_scalar("bt_lam", self.lambda_otf_bt, weight=0, round=4)
        
        # set the masked tokens for on-the-fly BT
        self.data_manager.update_step(num_updates)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)

        if update_num > 0:
            self.update_step(update_num)

        def weighted_loss_train_step(sample, weight):
            """Forward and backward step with weighted loss

            Args:
                sample (dict): training sample
                weight (float): loss weight
            """
            with torch.autograd.profiler.record_function("forward"):
                with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                    loss, sample_size, logging_output = criterion(model, sample)
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)
            return loss, sample_size, logging_output

        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, {}

        for task in self.args.train_tasks:
            if task not in sample:
                continue
            task_sample = sample[task]
            loss_weight = 0.0
            if task == BT_NAME:
                loss_weight = self.lambda_otf_bt
            elif task == DAE_NAME:
                loss_weight = self.lambda_denoising
            elif task == MT_NAME:
                loss_weight = 1.0
            
            if loss_weight == 0:
                continue

            examples_msg = "\n" + "=" * 20 + f"{task} examples" + "=" * 20
            for lang_pair in task_sample:
                lp_smp = task_sample[lang_pair]
                loss, sample_size, logging_output = weighted_loss_train_step(lp_smp, loss_weight)
                agg_loss += loss.detach().item()
                agg_sample_size += sample_size
                for k in logging_output:
                    if k not in agg_logging_output:
                        agg_logging_output[k] = 0.0
                    agg_logging_output[k] += logging_output[k]
                    
                    task_key=f"{task}_{k}"
                    if task_key not in agg_logging_output:
                        agg_logging_output[task_key] = 0.0
                    agg_logging_output[task_key] += logging_output[k]

                if self.args.print_train_samples:
                    src_lang, tgt_lang = lang_pair.split('-')
                    src_str, tgt_str = None, None
                    ln = lp_smp["net_input"]["src_tokens"].shape[0]
                    if ln > 0:
                        src_tokens = lp_smp["net_input"]["src_tokens"][-1]
                        tgt_tokens = lp_smp["target"][-1]
                        src_str = self.source_dictionary.string(src_tokens, self.args.bpe_symbol, include_eos=True)
                        tgt_str = self.target_dictionary.string(tgt_tokens, self.args.bpe_symbol, include_eos=True)

                    examples_msg += f"\n[{src_lang}] {src_str} \n[{tgt_lang}] {tgt_str}\n"

            if self.args.print_train_samples:
                logger.info(examples_msg)

        agg_logging_output[f"{BT_NAME}_lam"] = self.lambda_otf_bt
        agg_logging_output[f"{DAE_NAME}_lam"] = self.lambda_denoising
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.bleu_sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu
        from fairseq_cli.generate import get_symbols_to_strip_from_output
        
        def decode(toks, escape_unk=False):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator)
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=sample["target"][:,[0]])
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])