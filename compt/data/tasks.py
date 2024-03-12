from collections import OrderedDict
from datasets import Dataset
import collections
import abc
import os
import os.path as op
import pandas as pd
import functools
from pathlib import Path
from typing import Callable, List, Mapping
from utils import pad_punctuation
from metrics import metrics
from .utils import round_stsb_target, defdict
import datasets
import logging
import numpy as np
import torch
import re
from attempt.maps import *
import attempt.mylogs as mylogs
from itertools import cycle, islice
from random import shuffle

logger = logging.getLogger(__name__)

super_glue = mylogs.home + "/datasets/super_glue.py"

class AbstractTask(abc.ABC):
    name = NotImplemented
    do_shuffle = True # My code
    config = NotImplemented
    prefix = NotImplemented
    preprocessor: Callable = NotImplemented
    metric = NotImplemented
    metric_names = NotImplemented
    generation = False
    split_map = None
    do_split = False
    labels_list = None
    pcounter = 0
    rel_nat = None
    samples_per_head = 1
    map_labels = True
    labels_map = {"map":{}} # verbelizer
    split_to_data_name = {}
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
                                         "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
                                         "superglue-boolq", "xsum", "scitail"]
    large_data_without_all_splits = ["qqp", "qnli", "superglue-record", "sst2", "squad", "snli", "anli",
                                     "amazon_polarity", "yelp_polarity", "winogrande", "newsqa", "searchqa", "triviaqa", "nq", "hotpotqa"]

    def __init__(self, config, task_args, task="", tokenizer=None):
        self.config = config
        self.data_path = task_args.data_path
        self.seed = task_args.data_seed
        self.template = task_args.template
        self.tokenizer = tokenizer
        self.prefix = task_args.get("prefix", self.name)
        ## list of prompts
        if task: 
            self.task_name = task
        if not self.rel_nat:
            self.rel_nat = task
        self.rel_tok = "<" + task + ">"
        self.rel_word = task
        self.prompt_set = {} 
        prompt_config = {}
        self.mapping = task_args.mapping
        self.map_style = task_args.map_style
        if self.map_labels is True:
            self.labels_map["distinct"] = {}
            for i, label in enumerate(self.labels_list):
               self.labels_map["distinct"][label] = self.name + str(i)
        prompt_config["length"] = task_args.prompt_length
        prompt_config["target_length"] = task_args.target_prompt_length
        prompt_config["fixed_length"] = task_args.fixed_lenght_prompt
        self.multi_choice = task_args.multi_choice
        self.prompt_config = prompt_config
        self.task_args = task_args
        self.counter = {} #counter for logging items

    def get_id(self):
        return self.name 

    def get_max_target_length(self, tokenizer, default_max_length):
        ll = []
        if self.labels_list is not None:
           for label in self.labels_list:
              if self.mapping in self.labels_map and self.labels_map[self.mapping]:
                  label = self.labels_map[self.mapping][label]
              ll.append(len(tokenizer.encode(label))) 
           return max(ll) + 5
        return default_max_length

    def check_n_obs(self, n_obs, total_size):
        if n_obs < 0 or (n_obs is not None and n_obs > total_size):
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def shuffled_indices(self, dataset):
        if not self.do_shuffle:
            num_samples = len(dataset)
            return range(num_samples)
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        mylogs.bp("filter")
        if indices is None:
            indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        ds = dataset.select(indices)
        return ds

    def get_data_path(self, split):
        path = self.data_path
        if not path.startswith("/"):
            path= op.join(mylogs.home, self.data_path)
        if split in self.split_to_data_name: 
            ds_name = self.split_to_data_name[split] 
        else:
            ds_name = self.name
        path = op.join(path, ds_name)
        #if split == "test":
        #    path = op.join(path, self.config)
        Path(path).mkdir(parents=True, exist_ok=True)
        self.split = split
        file_path = op.join(path, split + '.tsv')
        return file_path

    def save_dataset(self, dataset, output_filename):
        if isinstance(dataset, pd.DataFrame):
            # Save Pandas DataFrame to CSV
            dataset.to_csv(output_filename, index=False)
            print(f"Dataset saved as CSV: {output_filename}")
        elif isinstance(dataset, Dataset):
            dataset.to_pandas().to_csv(output_filename, index=False)
        else:
            raise ValueError("Unsupported dataset type. Cannot save.")

    def load_dataset(self, split, n_obs = None):
        return datasets.load_dataset(self.name, self.config, split=split)

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def map_dataset(self, dataset, prefix):
        mylogs.bp("map")
        return dataset.map(functools.partial(self.preprocessor, prefix=prefix),
                           remove_columns=dataset.column_names,
                           load_from_cache_file=False)

    def get(self, split, prefix="", n_obs=None, split_validation_test=False, lang=None, file_name=None):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        self.split = split
        mylogs.bp("get")
        file_path = self.get_data_path(split)
        directory = os.path.dirname(file_path)
        fname = self.name + "_" + split 
        extension = ".csv" 
        obs_str = str(n_obs) if n_obs is not None and n_obs > 0 else "all"
        if split == "train":
            if obs_str != "all":
                outfile = os.path.join(directory, 
                    fname + "_" + str(self.seed) + "_" + obs_str + extension)
            else:
                outfile = os.path.join(directory, fname + "_" + obs_str + extension)
        else:
            outfile = os.path.join(directory, fname + "_" + obs_str + extension)

        if Path(outfile).is_file():
            file_name = outfile


        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            if lang is not None:
                dataset = self.load_dataset(split=mapped_split, lang_code=lang)
            if file_name is not None:
                #dataset = datasets.load_dataset(
                #    'csv', data_files={split:file_name})[split]
                df = pd.read_csv(file_name)
                #df.label = df.label.astype(int)
                dataset = Dataset.from_pandas(df)
            else:
                dataset = self.load_dataset(split=mapped_split)
                indices = self.get_split_indices(
                    split, dataset, validation_size=len(dataset)//2)
                dataset = self.subsample(dataset, n_obs, indices)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            if lang is not None:
                dataset = self.load_dataset(split="train", lang_code=lang)
            if file_name is not None:
                #dataset = datasets.load_dataset(
                #    'csv', data_files={split:file_name})[split]
                df = pd.read_csv(file_name)
                #df.label = df.label.astype(int)
                dataset = Dataset.from_pandas(df)
            else:
                dataset = self.load_dataset(split="train")
                indices = self.get_split_indices(
                    split, dataset, validation_size=1000)
                dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split[split]
            if lang is not None:
                dataset = self.load_dataset(split=mapped_split, lang_code=lang)

            mylogs.bp("get")
            if file_name is not None: # and split == "test":
                mylogs.minfo("------------- LOADING FROM FILE:" + self.name + " ----------")
                #dataset = datasets.load_dataset(
                #    'csv', data_files={split:file_name})[split]
                df = pd.read_csv(file_name)
                #df.label = df.label.astype(int)
                dataset = Dataset.from_pandas(df)
            else:
                mylogs.minfo("------------- LOADING Dataset :" + self.name + " ----------")
                dataset = self.load_dataset(split=mapped_split)
                if n_obs is not None:
                    dataset = self.subsample(dataset, n_obs)

        if not Path(outfile).is_file(): 
            self.save_dataset(dataset, outfile)
        return self.map_dataset(dataset, prefix)

    #### my post proc
    def post_process(self, preds, labels):
        _preds, _labels = preds, labels
        if self.labels_map and self.mapping in self.labels_map:
           d = self.labels_map[self.mapping]
           _preds, _labels = [], []
           keys = list(d.keys())
           values = list(d.values())
           for pp in preds:
               if pp in values:
                   _preds.append(keys[values.index(pp)])
               else:
                   _preds.append("-1")
           for ll in labels:
               if ll in values:
                   _labels.append(keys[values.index(ll)])
               else:
                   _labels.append(ll)
        return _preds, _labels

    ######### my template functions
    def fill_prompt(self, template, name, place_holder, plen = 0, num_holder="_i"):
        _pholder = place_holder
        place_holder = place_holder.replace("task", self.get_id())  
        place_holder = place_holder.replace("[", "<")  
        place_holder = place_holder.replace("]", ">")  
        while _pholder in template:
            if num_holder in _pholder:
                prompt = ""
                start = 0
                if num_holder == "_i":
                    start = self.pcounter
                for i in range(start, start + plen):
                    token = place_holder
                    if num_holder != "_1":
                        token = token.replace(num_holder, "_" + str(i))  
                    else:
                        token = token.replace(num_holder, "")  
                    prompt += " " + token
                    self.pcounter += 1
            else:
                prompt = place_holder
            prompt = prompt.strip()
            for token in prompt.split():
                if not name in self.prompt_set:
                    self.prompt_set[name] = []
                if not token in self.prompt_set[name]:
                    self.prompt_set[name].append(token)
            template = template.replace(_pholder,prompt, 1)
        return template

    def get_prompt_length(self, pnum = 0, is_target = False):
        mylogs.bp("plen")
        if is_target:
            tlength = self.prompt_config["target_length"]
            if tlength is None: return 0
            if type(tlength) == list:
                return tlength[pnum] if pnum < len(tlength) else tlength[-1]
            else:
                return tlength
        plength = self.prompt_config["length"]
        if plength is None: return 0
        if type(plength) == list:
            return plength[pnum] if pnum < len(plength) else plength[-1]
        else:
            return plength

    def fill_prompt_regex(self, template, regex):
        m = re.search(regex, template)
        pnum = 0
        self.pcounter = 0
        while m: 
            if len(m.groups()) == 2:
                name = m.groups()[0]
                emb = m.groups()[1]
                plen = 1
                if emb.isdigit():
                    plen = int(emb)
                num_holder = "_" + str(plen)
                if emb == "i":
                    plen = self.get_prompt_length(pnum) 
                    num_holder = "_i"
                elif emb == "j":
                    plen = self.get_prompt_length(pnum) 
                    num_holder = "_j"
                elif emb == "k":
                    plen = self.get_prompt_length(pnum, is_target=True) 
                    num_holder = "_k"
                place_holder = "[" + name + "_" + emb + "]"
                if "task" in name:
                    tid = self.get_id()
                    name = name.replace("task", tid)
                template = self.fill_prompt(template, name, place_holder, plen=plen, 
                        num_holder=num_holder)
                m = re.search(regex, template)
                pnum += 1
        return template

    def insert_prompts(self, template):
        mylogs.bp("fill_prompt")
        template = self.fill_prompt_regex(template, "\[([@a-zA-Z-]+)_(\d+)\]")
        template = self.fill_prompt_regex(template, "\[([@a-zA-Z\d-]+)_([a-zA-Z\?\d]+)\]")
        return template

    def get_prompts(self):
        data = {"task": self.get_id()}
        self.fill_template(data)
        return self.prompt_set

    def get_template_format(self):
        src = "(prefix) (prompt) (nat_prefix) {source} (prefix) (prompt) (nat) (prompt) (mask)" 
        target = "(mask) (prefix) (nat) {target}" # {end}"
        return src, target

    def get_template(self):
        src, target = self.get_template_format()
        parts = self.template.split("-")
        pcom = 0 # number of shared prompts among all tasks
        mylogs.bp("template")
        for part in parts:
            if part == "mask": 
               src = src.replace("(mask)", "{mask} (mask)")
               target = target.replace("(mask)","{mask} (mask)")
            elif part == "unsup": 
               src = src.replace("(mask)", "{mask}")
               target = target.replace("(mask)","{mask}")
            elif part == "unsupnat": 
               target = target.replace("(mask)","{mask}")
            elif part == "sup":
               src = src.replace("(mask)", "")
               target = target.replace("(mask)","")
            elif part == "pcom":
               src = src.replace("(prompt)", "[com_i] (prompt) ",1)
               pcom += 1
            elif part == "pmask":
               src = src.replace("(prompt)", "[tar-task_k] {mask} (prompt) ",1)
            elif part == "ptar":
               src = src.replace("(prompt)", "[tar-task_k] (prompt) ",1)
            elif part == "p0" or part == "0":
               src = src.replace("(prompt)", "",1)
            elif part == "px0" or part == "0":
               src = src.replace("(prefix)", "",1)
            elif part == "px":
               src = src.replace("(prefix)", "{prefix}",1)
            elif part == "pt":
               src = src.replace("(prompt)", "[task_i] (prompt) ",1)
            elif part == "pnat":
               src = src.replace("(prompt)", "{prompt_from_nat} (prompt) ",1)
            elif part == "pn":
               src = src.replace("(prompt)", "{prompt_n} (prompt) ",1)
            elif part == "pnt":
               src = src.replace("(prompt)", "{prompt_nt} (prompt) ",1)
            elif part == "pnr":
               src = src.replace("(prompt)", "{prompt_nr} (prompt) ",1)
            elif part == "psh":
               src = src.replace("(prompt)", "{prompt_shared_tokens} (prompt) ",1)
            elif part == "psht":
               src = src.replace("(prompt)", "{prompt_task_eq_shared} (prompt) ",1)
            elif part == "pshr":
               src = src.replace("(prompt)", "{prompt_shared_random} (prompt) ",1)
            elif part == "nat_prefix":
               src = src.replace("(nat_prefix)", "{rel_nat}", 1)
            elif part == "nat_input" or part == "nat": 
               src = src.replace("(nat)", "{rel_nat}", 1)
            elif part == "input_shared_words":
               src = src.replace("(prefix)", "{rel_shared_words}:", 1)
            elif part == "nat_target": 
               target = target.replace("(nat)", "{rel_nat}", 1)
            elif part == "target_shared_words": 
               target = target.replace("(prefix)", "{rel_shared_words}:", 1)
            else:
                raise ValueError("Invalid part in template:" + part)

        # remove unused place holders
        src = re.sub(r'\(.*?\)','',src)
        src = re.sub(' +', ' ',src)
        target = re.sub(r'\(.*?\)','',target)

        return src, target, pcom

    def extend_data(self, data, pcom=0):
        mylogs.bp("task")
        if "task" in data:
            task = data["task"]
            task = self.name
            data["rel_tok"] = REL_TO_TOKEN[task] if task in REL_TO_TOKEN else self.rel_tok
            data["rel_word"] = REL_TO_WORD[task] if task in REL_TO_WORD else self.rel_word
            data["rel_nat"] = REL_TO_PHRASE[task] if task in REL_TO_PHRASE else self.rel_nat
            rel_from_nat = REL_TO_PHRASE[task] if task in REL_TO_PHRASE else task
            rel_from_nat = rel_from_nat.split()
            num_prompts = self.task_args.setdefault("num_prompts",1)
            task_comb = self.task_args.setdefault("task_comb", "none")
            tid = self.task_args["id"]
            prompt_n = []
            if task_comb == "none":
                prompt_n = ["[p" + str(tid) + str(i) + "_i]" for i in range(num_prompts)]
            elif task_comb == "comb":
                prompt_n = ["[p" + str(ii) + "0_i]" for ii in range(1, tid + 1)]
                prompt_n.extend(["[p" + str(tid) + "_i]"])

            data["prompt_n"] = " ".join(prompt_n)
            shuffle(prompt_n)
            data["prompt_nr"] = " ".join(prompt_n)

            l = self.get_prompt_length(0)*(len(prompt_n) - pcom)
            prompt_nt = "[task" + "_" + str(l) + "]" 
            data["prompt_nt"] = prompt_nt

            prompt_from_nat = ["[task_" + w + "]" for w in rel_from_nat]
            prompt_from_nat_cycle = []
            for i in range(self.get_prompt_length(0)):
                j = i % len(rel_from_nat)
                tok = "[task" + "_" + rel_from_nat[j] + "?" + str(i) + "]"
                prompt_from_nat_cycle.append(tok)
            if self.prompt_config["fixed_length"]:
                data["prompt_from_nat"] = " ".join(prompt_from_nat_cycle)
            else:
                data["prompt_from_nat"] = " ".join(prompt_from_nat)

            if task in REL_TO_SHARED_TOKENS:
                rel_with_shared_tokens = REL_TO_SHARED_TOKENS[task] 
            else:
                rel_with_shared_tokens = task 
            rel_with_shared_tokens = rel_with_shared_tokens.split()
            data["rel_shared_words"] = " ".join(rel_with_shared_tokens)
            # prompt shr creates same prompts for shared tokens of tasks, 
            # the length of prompts 
            # is specified with i
            prompt_shared_tokens = ["[" + w + "_i]" for w in rel_with_shared_tokens]
            data["prompt_shared_tokens"] = " ".join(prompt_shared_tokens)
            # prompt is the same as prompt sh but the tokens are shuffled 
            shuffle(rel_with_shared_tokens)
            prompt_shared_random = ["[" + w + "_j]" for w in rel_with_shared_tokens]
            data["prompt_shared_random"] = " ".join(prompt_shared_random)
            # psht is for comparision. it uses task specific prompts with the length 
            # of shared prompts concatenated to each other, 
            # however prompts for each tasks are distnict
            # it also substract the length of common or shared prompts among all tasks
            l = self.get_prompt_length(0)*(len(rel_with_shared_tokens) - pcom)
            prompt_task_eq_shared = "[task" + "_" + str(l) + "]" 
            data["prompt_task_eq_shared"] = prompt_task_eq_shared

        return data

    def replace_mask(self, text):
        # Replace {mask} with <extra_id_i>
        mask_placeholder = "{mask}"
        mask_counter = 0
        while mask_placeholder in text:
            replacement = f"<extra_id_{mask_counter}>"
            text = text.replace(mask_placeholder, replacement, 1)
            mask_counter += 1
        return text

    def fill_template(self, data):
        mylogs.bp("fill")
        src,tgt,pcom = self.get_template()

        mask = "<extra_id_0>"
        data = self.extend_data(data, pcom=pcom)
        # data["mask"] = mask
        data["end"] = "</s>" 
        data["prefix"] = self.name + ":"
        data = defdict(data)
        # fill the templates with data

        # Replace masks in src and tgt
        # src_texts = src_texts.replace("{mask}", mask)
        src_texts = self.replace_mask(src).format_map(data)
        tgt_texts = self.replace_mask(tgt).format_map(data)

        src_texts = self.insert_prompts(src_texts)
        return src_texts, tgt_texts 

    def get_label_list(self):
        labels_list = []
        if self.labels_map and self.mapping:
            for label in self.labels_list:
                labels_list.append("<" + self.labels_map[self.mapping][label] + ">")
        return labels_list

    def seq2seq_format(self, sources: List[str],
                       targets: List[str],
                       prefix: str = None,
                       extra_fields={}):
        if not prefix:
            prefix = self.prefix
        if not prefix:
            prefix = self.name
        src_prefix = "src-" + prefix
        src_prefix += ":"
        mylogs.bp("format")
        mylogs.bp(self.split + "frm")
        if self.mapping in self.labels_map and self.labels_map[self.mapping]:
            labels_list = []
            for label in self.labels_list:
                labels_list.append(self.labels_map[self.mapping][label])

            tt = []
            for label in targets:
                assert label in self.labels_map[self.mapping], self.name + ":" + label \
                        + ":" + str(self.labels_map)
                # tt.append("<" + self.labels_map[label] + ">")
                tt.append(self.labels_map[self.mapping][label])
            targets = tt 
        else:
            labels_list = self.labels_list
            
        add_prefix = self.task_args.setdefault("add_prefix", False)
        orig_src = ' '.join(sources)
        sources = [src_prefix]+sources if add_prefix else sources
        src = ' '.join(sources)
        tgt =  ' '.join(targets)
        src = src.strip()
        tgt = tgt.strip()

        prompt_len = self.get_prompt_length()
        max_input_len = 511 - len(tgt) - prompt_len
        if self.multi_choice:
            max_input_len -= 9 # for options tag
            max_input_len -= sum([len(l) + 1 for l in labels_list])

        if self.multi_choice:
            src = src + " options:" + ",".join(labels_list)

        src = src[:max_input_len]

        data = {'source': src,
                'target': tgt,
                'task': self.get_id(),
                ** extra_fields}
        extra_fields = {}
        extra_fields["event"] = orig_src 
        extra_fields["tail"] = tgt 
        extra_fields["sel"] = False
        extra_fields["split"] = self.split 
        src_text, tgt_text = self.fill_template(data) 
        extra_fields["query"] = src_text
        extra_fields["resp"] = tgt_text
        extra_fields["target_text"] = tgt_text
        if not "examples" in self.counter:
            self.counter["examples"] = 1
        if self.counter["examples"] < 5:
            mylogs.vlog.info(f"=========== Extra Fields | split={self.split} =========")
            mylogs.vlog.info("%s", extra_fields)
            self.counter["examples"] += 1
        mylogs.bp("format")
        return {'source': src_text,
                'target': tgt_text, 
                'task': self.name,
                'extra_fields': extra_fields}

class Squad(AbstractTask):
    name = "squad"
    metric = [metrics.squad]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split)

    def preprocessor(self, example, prefix):
        answer = pad_punctuation(example['answers']).split("\t")
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question[:100],
                  "context:", context[:350]]
        target = [answer] if type(answer) == str else answer
        return self.seq2seq_format(source, target, prefix)



class DROP(AbstractTask):
    name = "drop"
    metric = [metrics.squad]

    def load_dataset(self, split):
        if split == "train":
            return datasets.load_dataset("json", field="history_690", 
                    data_files=op.join(
                        mylogs.home, "drop/drop_dataset/drop_dataset_train.json"))
        else:
            return datasets.load_dataset("json", field="history_690",
                    data_files=op.join(
                        mylogs.home, "drop/drop_dataset/drop_dataset_dev.json"))


    def preprocessor(self, example, prefix):
        answer = pad_punctuation(example['answers_spans']['spans'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['passage'])
        source = ["question:", question,
                  "context:", context]
        target = [answer]
        return self.seq2seq_format(source, target, prefix)


class PIQA(AbstractTask):
    name = "piqa"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    labels_map = {"map":{"0":"Choice1", "1":"Choice2", "0.0":"Choice1", "1.0":"Choice2"}}

    def load_dataset(self, split):
        return datasets.load_dataset('piqa', split=split)
        path = op.join(mylogs.home, "piqa","final", split + ".csv")
        # return datasets.load_dataset('csv', data_files=path)
        df = pd.read_csv(path)
        #df.label = df.label.astype(int)
        ds = Dataset.from_pandas(df)
        return ds

    def preprocessor(self, example, prefix):
        src_texts = ["question:", example['goal'], "choice1:",
                     example["sol1"][0], "choice2:", example["sol2"][0]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class CommonsenseQA(AbstractTask):
    name = "commonsense-qa"
    labels_list = ["0", "1", "2", "3", "4"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "test":"validation",
                           "validation": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('commonsense_qa', split=split)

    def preprocessor(self, example, prefix):
        label2id = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}
        src_texts = ["question:", example['question'], "choice1:", example["choices"]["text"][0], "choice2:", example["choices"]["text"][1],
                     "choice3:", example["choices"]["text"][2], "choice4:", example["choices"]["text"][3], "choice5:", example["choices"]["text"][4]]
        tgt_texts = [label2id[example["answerKey"]]]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SocialIQA(AbstractTask):
    name = "social-i-qa"
    labels_list = ["0", "1", "2"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "test":"validation",
                           "validation": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset(mylogs.home + '/datasets/social_i_qa.py', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["question:", example['question'], "context:", example["context"], "|| choice0:",
                     example["answerA"][0], "|| choice1:", example["answerB"][0], "|| choice2:", example["answerC"][0]]
        tgt_texts = [str(int(example["label"]) - 1)]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SciTail(AbstractTask):
    name = "scitail"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('scitail', "snli_format", split=split)

    def preprocessor(self, example, prefix):
        label2id = {"entailment": "0", "neutral": "1"}
        src_texts = ["premise:", example['sentence1'],
                     "hypothesis:", example["sentence2"]]
        tgt_texts = [label2id[example["gold_label"]]]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    #labels_map = {"map":{"0":"unequal","1":"duplicate"}
    labels_map = {
            "map":{"0":"not_equivalent","1":"equivalent"},
            "map2":{"0":"not_equal","1":"duplicate"}
            }
    #labels_map = {"map":{"0":"F","1":"G"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc', split=split) 

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class COLA(AbstractTask):
    name = "cola"
    labels_list = ["0", "1"]
    metric = [metrics.matthews_corrcoef]
    metric_names = ["matthews_correlation"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    #labels_map = {"map":{"0": "inadmissible", "1":"acceptable"}
    labels_map = {"map":{"0": "unacceptable", "1":"acceptable"}}
    #labels_map = {"map":{"0": "A", "1":"B"}
    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)

class IMDB(AbstractTask):
    name = "imdb"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "train",
                           "test": "test"}
    labels_map = {"map":{"0":"negative", "1":"positive"}}
    rel_nat = "The sentiment is "

    def load_dataset(self, split):
        return datasets.load_dataset('imdb', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class TweetEval(AbstractTask):
    name = "tweet-eval"
    labels_list = ["0", "1", "2"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    labels_map = {
            "map":{"0":"negative", "1":"neutral", "2":"positive"},
            }
    rel_nat = "The sentiment is"

    def load_dataset(self, split):
        return datasets.load_dataset('tweet_eval', 'sentiment',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)



class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    labels_map = {
            "map":{"0":"negative", "1":"positive"},
            }
    #labels_map = {"map":{"0":"bad", "1":"good"}
    # labels_map = {"map":{"0":"L", "1":"M"}
    #rel_nat = "As a result, they feel"
    rel_nat = "The sentiment is"

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class YelpPolarity(AbstractTask):
    name = "yelp_polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split):
        print(split)
        return datasets.load_dataset('yelp_polarity')[split]

    def preprocessor(self, example, prefix):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class Amazon_Polarity(AbstractTask):
    name = "amazon_polarity"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train", "test": "test"}

    def load_dataset(self, split):
        return datasets.load_dataset('yelp_polarity', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence:", "<title> {0} <context> {1}".format(
            example['title'], example['context'])]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class STSB(AbstractTask):
    name = "stsb"
    map_labels = False
    labels_list = [str(np.round(label, decimals=1))
                   for label in np.arange(0, 5.2, 0.2)]
    metric = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    metric_names = ["pearson", "spearmanr"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)

class Atomic(AbstractTask):
    name = "atomic"
    map_labels = False
    metric = [metrics.rouge]
    metric_names = ["rouge"]
    generation = True
    do_shuffle = True
    samples_per_head = 3
    rels = []
    split_to_data_name = {"train":"atomic", "test":"atomic"}
    def __init__(self, config, task_args, task="", tokenizer=None):
        super().__init__(config, task_args, task, tokenizer)
        if not task_args.rels:
            self.rels = [self.name]
        else:
            self.rels = task_args.rels

    def load_dataset(self, split):
        if split != "train" or self.do_split:
            self.do_shuffle = False
        path = self.get_data_path(split)
        df = pd.read_table(path)
        if self.do_split or (split == "test" and len(df) < 300):
            path = self.get_data_path("train")
            df = pd.read_table(path)
            if split == "test":
                df = df.tail(300)
            else:
                df = df.head(len(df) - 300)
        df = self.filter(df, split)
        df = self.preproc_df(df, split)
        assert len(df) > 0, "data frame is empty for " + split + " of " + self.name + " " + path
        df = self.postproc_df(df, split)
        assert len(df) > 0, "data frame is empty for " + split + " of " + self.name + " " + path
        
        ds = Dataset.from_pandas(df)
        self.df = df
        return ds

    def check_n_obs2(self, n_obs, total_size):
        if n_obs < 0:
            return total_size
        df = self.df
        lst = df['input_text'].value_counts()[:n_obs].index
        out = df[df['input_text'].isin(lst)]
        #m = pd.Series(range(0, n_obs), index=lst)
        #out = df[df['input_text'].isin(lst)].sort_values('input_text', key=lambda x: m[x])
        n_obs = len(out)
        return n_obs

    def subsample(self, dataset, n_obs=None, indices=None):
        mylogs.bp("filter")
        rows = []
        counter = {}
        df = self.df
        for idx, row in df.iterrows():
            if not row.input_text in counter:
                counter[row.input_text] = 0
            counter[row.input_text] += 1
            if counter[row.input_text] > self.samples_per_head:
                continue
            rows.append(row.to_dict())
            if len(counter) > n_obs:
                break
        self.df = pd.DataFrame(data=rows)
        ds = Dataset.from_pandas(self.df)
        return ds

    def postproc_df(self, df, split):
        df = df[df.prefix == self.name]
        return df

    def preproc_df(self, df, split):
        mylogs.bp("filter")
        df["freqs"] = df.groupby(['input_text'])['input_text'].transform('count')
        df['px'] = df[['input_text','prefix']].groupby(['input_text'])['prefix'].transform(lambda x: ','.join(x))
        df['px_count'] = df[['input_text','prefix']].groupby(['input_text'])['prefix'].transform('nunique')
        print("len df:", len(df))
        # df = df.groupby(["prefix", "input_text"]).head(self.samples_per_head)
        print("len new df:", len(df))
        sort_by = ["px_count","freqs","input_text", "prefix"] 
        if "sel" in df:
            sort_by = ["sel", "freqs","input_text", "prefix"] 
        df = df.sort_values(by=sort_by, ascending=False)
        i = 0
        for idx, row in df.iterrows():
            text = "{}   {}   {}".format(row.input_text, row.prefix, row.target_text)
            mylogs.success(text, log=False)
            i += 1
            if i > 100:
                break;
        return df


    def preproc_df2(self, df, split):
        df["freqs"] = df.groupby(['prefix','input_text'])['input_text'].transform('count')
        print("len df:", len(df))
        df = df.groupby(["prefix", "input_text"]).head(self.samples_per_head)
        print("len new df:", len(df))
        sort_by = ["freqs","input_text", "prefix"] 
        if "sel" in df:
            mylogs.bp("df")
            sort_by = ["sel", "freqs","input_text", "prefix"] 
        df = df.sort_values(by=sort_by, ascending=False)
        i = 0
        for idx, row in df.iterrows():
            text = "{}   {}   {}".format(row.input_text, row.prefix, row.target_text)
            mylogs.success(text)
            i += 1
            if i > 30:
                break;
        return df

    def filter(self, df, split):
        cond = ""
        mylogs.bp("filter")
        df = df[~df["target_text"].str.contains('none', na=False)]
        for val in self.rels: 
            cond += f"| (df['prefix'] == '{val}') "
        cond = cond.strip("|")
        if cond: df = df[eval(cond)]
        return df

    #### ppppppppppppppp 
    def preprocessor(self, example, prefix):
        mylogs.bp("task_prep")
        src_texts = [str(example["input_text"])]
        tgt_texts = [str(example["target_text"])]
        extra_fields = {}
        extra_fields["event"] = example["input_text"]
        extra_fields["rel"] = example["prefix"]
        extra_fields["tail"] = example["target_text"]
        extra_fields["sel"] = example["sel"] if "sel" in example else False
        return self.seq2seq_format(src_texts, tgt_texts, 
                prefix, extra_fields=extra_fields)

class xIntent(Atomic):
    name = "xIntent"

class isAfter(Atomic):
    name = "isAfter"

class isBefore(Atomic):
    name = "isBefore"

class AtomicRel(Atomic):
    name = "atomic-rels"
    samples_per_rel = 100
    def __init__(self, config, task_args, task=""):
        super().__init__(config, task_args)
        self.train_samples_per_rel = task_args.train_samples
        self.val_samples_per_rel = task_args.val_samples
        self.test_samples_per_rel = task_args.test_samples

    def get_id(self):
        return "-".join(self.rels)

    def preproc_df(self, df, split):
        if split == "train":
            samples_per_rel = self.train_samples_per_rel
        elif split == "validation":
            samples_per_rel = self.val_samples_per_rel
        else:
            samples_per_rel = self.test_samples_per_rel
        print("len df:", len(df))
        df = df.groupby(["prefix"]).head(samples_per_rel)
        print("len new df:", len(df))
        return df

    def check_n_obs(self, n_obs, total_size):
        return total_size

    def get_data_path(self, split):
        path = self.data_path
        self.split = split
        if not path.startswith("/"):
            path= op.join(mylogs.home, self.data_path)
        if split == "test":
            path = op.join(path, self.config, 'test.tsv')
        else:
            path = op.join(path, split + '.tsv')
        return path

    def preprocessor(self, example, prefix):
        src_texts = ["head:", str(example["input_text"]), 
                    "tail:", str(example["target_text"])]
        tgt_texts = [example["prefix"].strip()]
        extra_fields = {}
        extra_fields["event"] = example["input_text"]
        extra_fields["tail"] = example["target_text"]
        extra_fields["sel"] = example["sel"] if "sel" in example else False
        return self.seq2seq_format(src_texts, tgt_texts, 
                prefix, extra_fields=extra_fields)

class Causes(Atomic):
    name = "Causes"

class xReason(Atomic):
    name = "xReason"

class Desires(Atomic):
    do_split = True
    name = "Desires"

class xAttr(Atomic):
    name = "xAttr"

class xNeed(Atomic):
    name = "xNeed"

class xReact(Atomic):
    name = "xReact"

class oReact(Atomic):
    name = "oReact"

class AtLocation(Atomic):
    name = "AtLocation"
    rel_nat = "is located at"

class ObjectUse(Atomic):
    name = "ObjectUse"
    rel_nat = "is used for"

class Desires(Atomic):
    name = "Desires"
    rel_nat = "desire"

class CapableOf(Atomic):
    name = "CapableOf"
    rel_nat = "is capable of"

class HasProperty(Atomic):
    name = "HasProperty"
    rel_nat = " has the property of "

class isFilledBy(Atomic):
    name = "isFilledBy"
    rel_nat = "is filled by"

class xWant(Atomic):
    name = "xWant"

class oWant(Atomic):
    name = "oWant"

class xEffect(Atomic):
    name = "xEffect"

class oEffect(Atomic):
    name = "oEffect"


class CommonGen(AbstractTask):
    name = "common-gen"
    metric = [metrics.rouge]
    metric_names = ["rouge"]
    generation = True

    def load_dataset(self, split):
        return load_dataset(mylogs.home + "/datasets/common_gen.py", split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["concepts:"] + example["concepts"]
        tgt_texts = [example['target']]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class QQP(AbstractTask):
    name = "qqp"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score_with_invalid, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    labels_map = {
            "map1":{"0":"not_equivalent","1":"equivalent"},
            "map":{"0":"not_duplicate","1":"duplicate"},
            "map2":{"0":"not_equal","1":"duplicate"},
            "map3":{"0":"different","1":"duplicate"},
            }
    #labels_map = {"map":{"0":"unequal","1":"duplicate"}
    #labels_map = {"map":{"0":"different","1":"identical"}
    #labels_map = {"map":{"0":"F","1":"G"}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['question1'],
                     "sentence2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    # labels_map = {"map":{"0":"en", "1":"neutral", "2": "contradicts"}
    labels_map = {
            "map":{"0":"entailment", "1":"neutral", "2": "contradiction"},
            "map2":{"0":"entailment", "1":"neutral", "2": "contradiction"}
            }
    # labels_map = {"map":{"0":"0", "1":"1", "2": "2"}
    # labels_map = {"map":{"0":"C", "1":"D", "2": "E"}
    rel_nat = "The logical relation between premise and hypothesis is " 

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['premise'], 
                     "sentence2:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)

class ParsNLI(AbstractTask):
    name = "parsnli"
    labels_list = ["c", "e", "n"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    #labels_map = {"map":{"e":"en", "n":"neutral", "c": "contradiction"}

    def load_dataset(self, split):
        return datasets.load_dataset("persiannlp/parsinlu_entailment", split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example['sent1'],
                     "hypothesis:", example["sent2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)

class PAWS(AbstractTask):
    name = "paws"
    labels_list = ["0", "1"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {
            "map":{"0":"not_equivalent","1":"equivalent"}
            }

    def load_dataset(self, split):
        return datasets.load_dataset("paws", "labeled_final", split=split)
        return datasets.load_dataset(mylogs.home + '/paws/paws.py', 
                'labeled_final', split=split)
        path = op.join(mylogs.home,"paws", "final", split + ".tsv") 
        df = pd.read_table(path)
        ds = Dataset.from_pandas(df)
        return ds

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SNLI(AbstractTask):
    name = "snli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {
            "map":{"0":"entailment", "1":"neutral", "2": "contradiction"}
            }

    def load_dataset(self, split):
        return datasets.load_dataset('snli', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example['premise'],
                     "hypothesis: ", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class MultiNLI(AbstractTask):
    name = "multinli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {
            "map":{"0":"entailment", "1":"neutral", "2": "contradiction"}
            }

    def load_dataset(self, split):
        return datasets.load_dataset('multi_nli', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example['premise'],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    #rel_nat = "Can the question be answered by the passage?"
    rel_nat = "The logical relation between sentence and question is "
    labels_map = {"map":{"0":"entailment", "1":"not_entailment"}}
    #labels_map = {"map":{"0":"entails", "1":"irrelated"}
    #labels_map = {"map":{"0":"yes", "1":"no"}
    #labels_map = {"map":{"0":"C", "1":"D"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['question'][:100],
                "sentence2:", example["sentence"][:350]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class RTE(AbstractTask):
    name = "rte"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    labels_map = {
            "map":{"0":"entailment", "1":"not_entailment"},
            "map2":{"0":"entailment", "1":"not_entailment"}
            } # entailment nont_entailment
    ## labels_map = {"map":{"0":"C", "1":"D"} # entailment nont_entailment
    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte',
                                     split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    labels_map = {"map":{"0":"not_entailment", "1":"entailment"}}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUEBoolQ(AbstractTask):
    name = "superglue-boolq"
    labels_list = ['0', '1']
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    labels_map = {"map":{"0":"False", "1":"True"}}
    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'boolq', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["question:", example["question"],
                     "passage:", example["passage"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUERTE(AbstractTask):
    name = "superglue-rte"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {"map":{"0":"entailment", "1":"not_entailment"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'rte', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUECB(AbstractTask):
    name = "superglue-cb"
    labels_list = ['0', '1', '2']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy]
    metric_names = ["f1_multiclass", "accuracy"]
    labels_map = {"map":{"0":"entailment", "2":"neutral", "1": "contradiction"}}
    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'cb', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUECOPA(AbstractTask):
    name = "superglue-copa"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {"map":{"0":"Choice1", "1":"Choice2"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'copa', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["premise:", example["premise"],
                     "choice1:", example["choice1"],
                     "choice2:", example["choice2"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUEMultiRC(AbstractTask):
    name = "superglue-multirc"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.multirc_f1_over_all_answers,
              metrics.mean_group_metric(metrics.exact_match)]
    metric_names = ["f1", "em"]
    labels_map = {"map":{"0":"False", "1":"True"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'multirc', split=split)

    def remove_markup(self, text):
        """Removes the HTML markup."""
        text = re.sub('<br>', ' ', text)
        text = re.sub('<(/)?b>', '', text)
        return text

    def preprocessor(self, example, prefix):
        group = example['idx']['question']
        # T5 applies remove_markup to the joined string, but this should not make
        # any difference as well.
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/data/preprocessors.py#L797
        src_texts = ["question:", self.remove_markup(example["question"]),
                     "answer:", self.remove_markup(example["answer"]),
                     "paragraph:", self.remove_markup(example["paragraph"])]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix, extra_fields={"group": group})


class SuperGLUEWIC(AbstractTask):
    name = "superglue-wic"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {"map":{"0":"False", "1":"True"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'wic', split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence1:", example["sentence1"],
                     "sentence2:", example["sentence2"],
                     "word:", example["word"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)

from datasets import load_dataset, DownloadConfig

download_config = DownloadConfig(
        proxies={
            "http": "http://fodev.org:8118",
            "https": "http://fodev.org:8118"
            }
        )
class SuperGLUEWSCFixed(AbstractTask):
    # source: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
    """Convert WSC examples to text2text format.
     WSC includes a sentence along with 2 'spans': the first denoting a noun and
     the other a pronoun. The 'label' specifies whether or not the pronoun is
     referencing the noun. This preprocessor puts ' * ' around the noun and ' # '
     around the pronoun.
     For example, a typical example from WSC might look like
     {
         'text': 'This is a test sentence .',
         'span1_text': 'test',
         'span1_index': 3,
         'span2_text': 'This',
         'span2_index': 0,
         'label': 0
     }
     This example would be transformed to
     {
         'inputs': 'wsc text: # This # is a * test * sentence .',
         'targets': 'False'
     }
    """
    name = "superglue-wsc.fixed"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    labels_map = {"map":{"0":"False", "1":"True"}}

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 
                'wsc.fixed', split=split)
        #, download_config=download_config)

    def _mark_span(self, text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub('N', str(span_idx), pattern_tmpl)
        pattern = re.sub('W', span_str, pattern)
        return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

    def preprocessor(self, example, prefix):
        # converts text as done in T5.
        text = example['text']
        text = self._mark_span(
            text, example['span1_text'], example['span1_index'], '*')
        # Compensate for 2 added "words" added in previous step.
        span2_index = example['span2_index'] + 2 * \
            int(example['span1_index'] < example['span2_index'])
        text = self._mark_span(text, example['span2_text'], span2_index, '#')
        src_texts = ["text:", text]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


class SuperGLUERecord(AbstractTask):
    """Convert ReCoRD examples to text2text examples.
    ReCoRD contains a passage, query containing a '@placeholder' string, and a set
    of entities that are the possible values of the placeholder. Each train and
    validation example will have a list of answers, any of which would be
    considered correct.
    For example, a typical example from ReCoRD might look like
    {
      'passsage': 'This is the passage.',
      'query': 'A @placeholder is a bird.',
      'entities': ['penguin', 'potato', 'pigeon'],
      'answers': ['penguin', 'pigeon'],
    }
    which this preprocessor would turn into the following two examples:
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'penguin',
    }
    and
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'pigeon',
    }
    """
    name = "superglue-record"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.squad]
    metric_names = ["squad"]

    def load_dataset(self, split):
        return datasets.load_dataset(super_glue, 'record', split=split)

    def preprocessor(self, batch, prefix):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # updates the passage.
            passage = ex['passage']
            passage = re.sub(
                r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            if prefix:
                inputs = self.name + " " + inputs
            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            answers = ex["answers"] if num_answers > 0 else ["<unk>"]
            for ans in answers:
                fmt = self.seq2seq_format([inputs],[ans], prefix)
                new_batch["source"].extend([fmt["source"]])
                new_batch["target"].extend([fmt["target"]])
                new_batch["task"].extend([self.name])
                exf = {**fmt["extra_fields"], **{"answers": ex["answers"]}}
                new_batch["extra_fields"].extend([exf])
        return new_batch

    def map_dataset(self, dataset, prefix):
        return dataset.map(functools.partial(self.preprocessor, prefix=prefix),
                           batched=True, remove_columns=dataset.column_names)

class WinoGrande(AbstractTask):
    name = "winogrande"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        return datasets.load_dataset('winogrande', "winogrande_xl", split=split)

    def preprocessor(self, example, prefix):
        src_texts = ["sentence:", example["sentence"],
                     "option0:", example["option1"],
                     "option1:", example["option1"]]
        tgt_texts = [str(int(example["answer"]) - 1)]
        return self.seq2seq_format(src_texts, tgt_texts, prefix)


TASK_MAPPING = OrderedDict(
    [
        ('atomic', Atomic),
        ('isAfter', isAfter),
        ('isBefore', isBefore),
        ('xIntent', xIntent),
        ('xReason', xReason),
        ('Desires', Desires),
        ('Causes', Causes),
        ('xAttr', xAttr),
        ('xNeed', xNeed),
        ('xReact', xReact),
        ('oReact', oReact),
        ('AtLocation', AtLocation),
        ('ObjectUse', ObjectUse),
        ('Desires', Desires),
        ('CapableOf', CapableOf),
        ('HasProperty', HasProperty),
        ('isFilledBy', isFilledBy),
        ('xWant', xWant),
        ('oWant', oWant),
        ('xEffect', xEffect),
        ('oEffect', oEffect),
        ('atomic-rels', AtomicRel),
        ('squad', Squad),
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('tweet-eval', TweetEval),
        ('imdb', IMDB),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('parsnli', ParsNLI),
        ('qqp', QQP),
        ('stsb', STSB),
        ('superglue-boolq', SuperGLUEBoolQ),
        ('superglue-rte', SuperGLUERTE),
        ('superglue-cb', SuperGLUECB),
        ('superglue-copa', SuperGLUECOPA),
        ('superglue-multirc', SuperGLUEMultiRC),
        ('superglue-wic', SuperGLUEWIC),
        ('superglue-wsc.fixed', SuperGLUEWSCFixed),
        ('superglue-record', SuperGLUERecord),
        ('multinli', MultiNLI),
        ('snli', SNLI),
        ('piqa', PIQA),
        ('drop', DROP),
        ('newsqa', Squad),
        ('searchqa', Squad),
        ('triviaqa', Squad),
        ('nq', Squad),
        ('hotpotqa', Squad),
        ("social-i-qa", SocialIQA),
        ("commonsense-qa", CommonsenseQA),
        ("common-gen", CommonGen),
        ("winogrande", WinoGrande),
        ("scitail", SciTail),
        ('yelp_polarity', YelpPolarity),
        ('amazon_polarity', Amazon_Polarity),
        ('paws', PAWS),
    ]
)


class AutoTask:
    @classmethod
    def get(self, task, config, task_args=None, tokenizer=None):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, task_args, task, tokenizer)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model.\n" + \
            "Task name should be one of {}.".format(task,
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
