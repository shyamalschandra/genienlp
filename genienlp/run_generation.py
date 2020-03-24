# Parts of this file were adopted from https://github.com/huggingface/transformers.
# See the original copyright notice below.

# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from tqdm import trange, tqdm
import math
import json
import csv
import re
import copy
import numpy as np

# multiprocessing with CUDA
from torch.multiprocessing import Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import torch
import torch.nn.functional as F

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig, BertConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer
from transformers import BertForMaskedLM, BertTokenizer

from .util import set_seed, get_number_of_lines, combine_files_on_disk, split_file_on_disk, get_file_part_path, detokenize, top_k_top_p_filtering
from .metrics import computeBLEU


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig, BertConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
    'bert': (BertForMaskedLM, BertTokenizer),
}


def apply_repetition_penalty(logits, context, repetition_penalty, prompt_token_id, pad_token_id):
    """ repetition penalty from CTRL (https://arxiv.org/abs/1909.05858), but much faster on GPU
        we penalize only the tokens that appear in the context, not in the generated text
    """
    m = torch.scatter(input=torch.zeros_like(logits), dim=1, index=context, value=1)
    m[:prompt_token_id] = 0
    m[:pad_token_id] = 0
    # print('m = ', m.shape)
    need_change = m * logits
    need_divide = need_change > 0
    need_multiply = need_change < 0
    logits = need_divide * logits / repetition_penalty + need_multiply * logits * repetition_penalty + (1-m) * logits
    
    # Old, slow implementation
    # if repetition_penalty != 1.0:
        # for i in range(context.shape[0]):
            # for _ in set(generated[i].tolist()):
                # if logits[i, _] > 0:
                    # logits[i, _] /= repetition_penalty
                # else:
                    # logits[i, _] *= repetition_penalty
    return logits

def sample_sequence(model, length, context, num_samples,
                    temperature=1.0, top_k=0, top_p=1.0, copy=0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu',
                    stop_token_ids=None, pad_token_id=None, supports_past=False, prompt_token_id=None, segment_token_ids=None,
                    start_reverse_position_ids=None, output_form=None):
    """
    Generates sequence of tokens for the batch of input contexts.
    Inputs:
        context: a list of token_ids, sorted by length from longest to shortest
        num_samples: the number of sequences to output for each input context
        length: The maximum length of generation in addition to the original sentence's length
        stop_token_ids: generation of each sequence will stop if we generate any of these tokens
        supports_past: set to True if the model accepts the 'past' input for more efficient generation. For example, GPT-2/Transfo-XL/XLNet/CTRL do
        segment_token_ids: a list of two integers that indicate the tokens we should use for each of the two segments
    """
    max_length = len(context[0]) # context is sorted by length from longest to shortest
    min_length = len(context[-1])

    # should not change the elements of context since it will change them outside this function as well.
    padded_context = []
    for i in range(len(context)):
        c = min(copy, len(context[i])-1) # -1 so that we do not copy prompt_token
        padded_context.append(context[i] + context[i][:c] + ([pad_token_id] * (max_length-len(context[i])+copy-c)))
    
    next_index = min_length + min(copy, min_length-1)
    length = max_length + (max_length - next_index) + length # generate till max_length, then generate another max_length+length tokens
    segment_ids = []

    completed_position_ids = []
    for i in range(len(context)):
        p = list(range(len(context[i])))
        segment_ids.append([segment_token_ids[0]]*len(p) + [segment_token_ids[1]]*(length+next_index-len(p)))
        if start_reverse_position_ids is None:
            completed_position_ids.append(p + list(range(length + next_index - len(p))))
        else:
            completed_position_ids.append(p + list(reversed(range(start_reverse_position_ids+len(p)))) + [0]*(length + next_index-start_reverse_position_ids-2*len(p)))

    position_ids = torch.tensor(completed_position_ids, dtype=torch.long, device=device)
    position_ids = position_ids.repeat(num_samples, 1)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)
    segment_ids = segment_ids.repeat(num_samples, 1)

    # print('context = ', context)
    # print('position_ids = ', position_ids)
    # print('segment_ids = ', segment_ids)

    context = torch.tensor(padded_context, dtype=torch.long, device=device)
    context = context.repeat(num_samples, 1)
    generated = context[:, :next_index]
    should_finish = None
    generated_logits = None
    past = None
    next_token = None
    with torch.no_grad():
        # rep_penalty = np.random.random(length) < 0.1
        # original_rep_penalty = repetition_penalty
        # print('rep_penalty = ', rep_penalty)
        for _ in range(length):
            inputs = {'input_ids': generated, 'position_ids': position_ids[:, :next_index], 'token_type_ids': segment_ids[:, :next_index]}
            if is_xlnet: 
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            if is_xlm_mlm and xlm_mask_token:
                # XLM MLM models are direct models (predict same token, not next token)
                # => need one additional dummy token in the input (will be masked and guessed)
                input_ids = torch.cat((generated, torch.full((1, 1), xlm_mask_token, dtype=torch.long, device=device)), dim=1)
                inputs = {'input_ids': input_ids}

            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            if supports_past:
                inputs['past'] = past
                if past is not None:
                    inputs['input_ids'] = next_token
                    inputs['position_ids'] = position_ids[:, next_index-1]
                    inputs['token_type_ids'] = segment_ids[:, next_index-1]
            
            outputs = model(**inputs)
            original_next_token_logits = outputs[0][:, -1, :]
            next_token_logits = original_next_token_logits / (temperature if temperature > 0 else 1.)
            past = outputs[1]

            next_token_logits = apply_repetition_penalty(next_token_logits, context, repetition_penalty,
                                                         prompt_token_id=prompt_token_id, pad_token_id=pad_token_id)
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            
            if output_form == 'logprob':
                generated_token_logit = F.log_softmax(original_next_token_logits, dim=-1).gather(1, next_token)
            else:
                assert output_form == 'logit'
                generated_token_logit = original_next_token_logits.gather(1, next_token)

            # throw away the tokens that we already have from the context
            if next_index < context.shape[1]:
                m = (context[:, next_index:next_index+1] != pad_token_id).long() # m==0 is where next_token should be kept
                next_token = m*context[:, next_index:next_index+1]+(1-m)*next_token
                generated_token_logit = (1-m)*generated_token_logit
            else:
                m = torch.zeros(1, device=device)

            for stop_token_id in stop_token_ids:
                if should_finish is None:
                    should_finish = ((next_token == stop_token_id) & (1-m).bool())
                else:
                    should_finish = should_finish | ((next_token == stop_token_id) & (1-m).bool())
            next_index += 1
            generated = torch.cat((generated, next_token), dim=1)
            if generated_logits is None:
                generated_logits = generated_token_logit
            else:
                generated_logits = torch.cat((generated_logits, generated_token_logit), dim=1)
            if should_finish.all():
                break
    return generated, generated_logits


special_token_mapping = {
    'PATH_NAME_0': {'forward': 'my1folder'},
    'PATH_NAME_1': {'forward': 'my2folder'},
    'TIME_0': {'forward': '1p.m.', 'back': ['1 pm', '1pm', '1:00 pm', '1:00pm', '1p.m.', '1 p.m.', '1:00 p.m.', '1:00']},
    'TIME_1': {'forward': '2p.m.', 'back': ['2 pm', '2pm', '2:00 pm', '2:00pm', '2p.m.', '2 p.m.', '2:00 p.m.', '2:00']},
    'EMAIL_ADDRESS_0': {'forward': 'e1@example.com'},
    'EMAIL_ADDRESS_1': {'forward': 'e2@example.com'},
    'URL_0': {'forward': 'my1site.com'},
    'URL_1': {'forward': 'my2site.com'},
    'DATE_0': {'forward': '5-6-2015', 'back': ['5-6-2015']},
    'DATE_1': {'forward': '8-3-2016', 'back': ['8-3-2016']},
    'CURRENCY_0': {'forward': '$12', 'back': ['$12', 'twelve dollars', '12 dollars', '$ 12', '$ 12.00', '12.00', '12']},
    'CURRENCY_1': {'forward': '$13', 'back': ['$13', 'thirteen dollars', '13 dollars', '$ 13', '$ 13.00', '13.00', '13']},
    'NUMBER_0': {'forward': '2', 'back': ['2', 'two']},
    'NUMBER_1': {'forward': '3', 'back': ['3', 'three']},
    'DURATION_0': {'forward': '5 weeks', 'back': ['5 weeks', 'five weeks']},
    'DURATION_1': {'forward': '6 weeks', 'back': ['6 weeks', 'six weeks']},
    'LOCATION_0': {'forward': 'locatio1n', 'back': ['locatio1n', 'locat1n']},
    'LOCATION_1': {'forward': 'locatio2n', 'back': ['locatio2n', 'locat2n']},
    'PHONE_NUMBER_0': {'forward': '888-8888'},
    'PHONE_NUMBER_1': {'forward': '777-8888'}
}

def create_features_from_tsv_file(file_path, tokenizer, input_column, gold_column, prompt_column, prompt_token, skip_heuristics):
    """
    Read a tsv file (this includes a text file with one example per line) and returns input features that the model needs
    """

    all_contexts = []
    all_context_tokens = []
    all_context_lengths = []
    all_golds = []
    reverse_maps = []

    number_of_lines = get_number_of_lines(file_path)
    with open(file_path) as input_file:
        reader = csv.reader(input_file, delimiter='\t')
        for row in tqdm(reader, desc='Reading Input File', total=number_of_lines):
            raw_text = row[input_column]
            all_golds.append(row[gold_column])
            # print('before text = ', raw_text)
            if skip_heuristics:
                reverse_maps.append({})
            else:
                raw_text, reverse_map = input_heuristics(raw_text)
                reverse_maps.append(reverse_map)
            all_contexts.append(raw_text)
            raw_text += prompt_token
            if prompt_column is not None and len(row) > prompt_column:
                raw_text += row[prompt_column]
            context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
            all_context_tokens.append(context_tokens)
            all_context_lengths.append(len(context_tokens))
            # if args.model_type == "ctrl":
                # if not any(context_tokens[0] == x for x in tokenizer.control_codes.values()):
                    # logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return all_contexts, all_context_tokens, all_context_lengths, all_golds, reverse_maps

def input_heuristics(s: str):
    """
    Changes the input string so that it is closer to what the pre-trained language models have seen during their training.
    Outputs:
        s: the new string
        reverse_map: a list of special tokens. Can be used to recover the original special_tokens in the string
    """
    reverse_map = []
    s = s.strip()
    s = detokenize(s)

    # Put question mark at the end whenever necessary.
    if s.startswith('which') or s.startswith('what') or s.startswith('where') or s.startswith('how') or s.startswith('who') or s.startswith('when'):
        if s.endswith('.'):
            s = s[:-1]
        if s[-1] != '?':
            s += '?'

    # replace special tokens with natural-looking exmaples
    sorted_special_token_mapping = sorted(special_token_mapping.items(), key=lambda x:len(x[0]), reverse=True) # sort to alwways start matching from the longest string
    for special_token, natural_form in sorted_special_token_mapping:
        new_s = s.replace(special_token, natural_form['forward'])
        if new_s != s:
            # print(new_s)
            reverse_map.append(special_token)
        s = new_s
    return s, reverse_map

def output_heuristics(s: str, reverse_map: list):
    for special_token in reverse_map:
        if 'back' in special_token_mapping[special_token]:
            back = special_token_mapping[special_token]['back']
        else:
            back = [special_token_mapping[special_token]['forward']]
        back = sorted(back, key=lambda x:len(x), reverse=True)
        for b in back:
            if b in s:
                s = s.replace(b, special_token)
                break
    return s

def lower_case(string):
    exceptions = [match.group(0) for match in re.finditer('[A-Z]+_[0-9]+', string)]
    for e in exceptions:
        string = string.replace(e, '<temp>', 1)
    string = string.lower()
    for e in exceptions:
        string = string.replace('<temp>', e, 1)

    return string

def compute_metrics(generations, golds):
    """
    Inputs:
        generations: a list of list of strings; generations[i] is a list of all generated outputs of the model for example i
        golds: a list of strings; golds[i] is the gold answer for example i
    """
    total_bleu = 0.0
    all_bleu = []
    exact_match = 0
    count = 0
    for idx, output in enumerate(generations):
        for sample in output:
            if re.sub('\s+', '', sample) == re.sub('\s+', '', golds[idx]):
                exact_match += 1
            bleu_score = computeBLEU([sample], [[golds[idx]]])
            all_bleu.append(bleu_score)
            total_bleu += bleu_score
            count += 1

    # from matplotlib import pyplot as plt
    # import numpy as np
    # h, b = np.histogram(all_bleu, bins=list(range(0, 105, 5)))
    # print('all_bleu = ', all_bleu)
    # print('h = ', h)
    # print('b = ', b)
    # h = h / np.sum(h)
    # print('h = ', h)
    # plt.title('GPT2 (temp=0, penalty=1.0) Paraphrases for restaurants')
    # plt.xlabel('BLEU with original')
    # plt.ylim((0.0, 1.0))
    # center = (b[:-1] + b[1:]) / 2
    # plt.bar(center, h, align='center', width=(b[1]-b[0]))
    # plt.savefig('./fig.png')

    return {'bleu': total_bleu/count, 'em': exact_match/count*100}

def parse_argv(parser):
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--input_file", type=str, help="The file from which we read prompts.")
    parser.add_argument('--input_column', type=int, required=True,
                        help='The column in the input file which contains the input sentences.')
    parser.add_argument('--prompt_column', type=int, default=None,
                        help='The column in the input file which contains the text we should start generation from.')
    parser.add_argument('--gold_column', type=int, default=None,
                        help='The column in the input file which contains the gold sentences. Defaults to --input_column if no gold is available.')
    parser.add_argument("--output_file", type=str, help="When specified, generated text will be written in this file.")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=20, help='The generated sentences will have a maximum length of len(input) + arg.length')
    parser.add_argument("--skip_heuristics", action='store_true', help='If True, will not replace special word such as NUMBER_0 in the input.')
    parser.add_argument("--do_lower_case", action='store_true', help='If True, will convert the output to lowercase. Has no effect if the model is already uncased.')
    
    # These can be used for improving the quality of the output
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--selection_criterion", type=str, choices=['none', 'average_logit', 'average_logprob', 'bleu'], default='none',
                        help='Select one of --num_sample outputs that maximizes this criterion')

    # These are generation hyperparameters. Each one can be a list of values in which case, we generate num_samples outputs for each set of hyperparameters.
    parser.add_argument("--start_reverse_position_ids", type=int, nargs='+', default=[None],
                        help='If provided, position ids will be the number of tokens left in generation and will start from len(input) + args.start_reverse_position_ids')
    parser.add_argument("--temperature", type=float, nargs='+', default=[1.0],
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, nargs='+', default=[1.0],
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, nargs='+', default=[0], help='0 disables top-k filtering')
    parser.add_argument("--top_p", type=float, nargs='+', default=[0.9], help='1.0 disables top-p filtering')
    parser.add_argument("--copy", type=int, nargs='+', default=[0],
                        help='Number of tokens that will be copied at the beginning of generation. Helps preserve the original meaning of the input sequence.')

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--prompt_token', type=str, default='<paraphrase>',
                        help="Token after which text generation starts. We add this to the end of all inputs.")
    parser.add_argument('--stop_tokens', type=str, nargs='+', default=['</paraphrase>', '?'],
                        help="Token at which text generation is stopped. The first element of the list is used as segment id as well.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for text generation for each GPU.")

def main(args):
    if args.prompt_column is not None and args.copy is not None and args.copy[0] != 0:
        raise ValueError('Cannot copy from the input and use prompt at the same time. Disable either --copy or --prompt_column.')
    hyperparameters = ['temperature', 'top_k', 'top_p', 'repetition_penalty', 'start_reverse_position_ids', 'copy']
    max_hyperparameter_len = max([len(getattr(args, h)) for h in hyperparameters])
    valid_len = [1, max_hyperparameter_len]
    for h in hyperparameters:
        if (len(getattr(args, h)) not in valid_len):
            logger.error('Hyperparameters should either have the same number of values as others or have exactly one value.')
        # If only one value is provided, use the same value for all samples
        setattr(args, h, getattr(args, h) * (max_hyperparameter_len // len(getattr(args, h))))

    logger.info('Will output %d sequences for each input.', args.batch_size*max_hyperparameter_len*args.num_samples)
    logger.info('Effective batch size for each GPU is %d', args.batch_size*args.num_samples)

    if args.gold_column is None:
        args.gold_column = args.input_column
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)
    args.model_type = args.model_type.lower()

    if args.n_gpu > 1:
        # Independent multi-GPU generation
        all_processes = []
        all_input_files = split_file_on_disk(args.input_file, args.n_gpu)
        for gpu_idx in range(args.n_gpu):
            copy_args = copy.copy(args)
            if torch.cuda.is_available() and not args.no_cuda:
                copy_args.device = torch.device("cuda:" + str(gpu_idx))
            copy_args.n_gpu = 1
            copy_args.input_file = all_input_files[gpu_idx]
            copy_args.output_file = get_file_part_path(args.output_file, gpu_idx)
            
            p = Process(target=run_generation, args=(copy_args,))
            all_processes.append(p)
            p.start()

        for p in all_processes:
            p.join()

        combine_files_on_disk(args.output_file, args.n_gpu, delete=True)

    else:
        run_generation(args)


def run_generation(args):
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)
    if args.model_type in ["ctrl"]:
        if args.temperature > 0.7:
            logger.info('CTRL typically works better with lower temperatures (and lower top_k).')

    xlm_lang = None
    # XLM Language usage detailed in the issues #1414
    if args.model_type in ["xlm"] and hasattr(tokenizer, 'lang2id') and hasattr(model.config, 'use_lang_emb') \
            and model.config.use_lang_emb:
        if args.xlm_lang:
            language = args.xlm_lang
        else:
            language = None
            while language not in tokenizer.lang2id.keys():
                language = input("Using XLM. Select language in " + str(list(tokenizer.lang2id.keys())) + " >>> ")
        xlm_lang = tokenizer.lang2id[language]

    # XLM masked-language modeling (MLM) models need masked token (see details in sample_sequence)
    is_xlm_mlm = args.model_type in ["xlm"] and 'mlm' in args.model_name_or_path
    if is_xlm_mlm:
        xlm_mask_token = tokenizer.mask_token_id
    else:
        xlm_mask_token = None

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    prompt_token_id = tokenizer.convert_tokens_to_ids(args.prompt_token)
    if pad_token_id is None:
        logger.error('Your tokenizer does not have a padding token')

    all_contexts, all_context_tokens, all_context_lengths, all_golds, reverse_maps = \
                                  create_features_from_tsv_file(file_path=args.input_file, tokenizer=tokenizer,
                                  input_column=args.input_column, gold_column=args.gold_column, prompt_column=args.prompt_column,
                                  prompt_token=args.prompt_token, skip_heuristics=args.skip_heuristics)

    
    # sort contexts based on their length so that less generated tokens are thrown away and generation can be done faster
    t = list(zip(*sorted(list(zip(all_context_lengths, all_contexts, all_context_tokens, range(len(all_context_tokens)), reverse_maps)), reverse=True)))
    all_context_lengths, all_contexts, all_context_tokens, original_order, reverse_maps = list(t[0]), list(t[1]), list(t[2]), list(t[3]), list(t[4])
    all_outputs = []

    if args.output_file is not None:
        output_file = open(args.output_file, 'w')

    stop_token_ids = [tokenizer.convert_tokens_to_ids(stop_token) for stop_token in args.stop_tokens]

    for batch in trange(math.ceil(len(all_context_tokens) / args.batch_size), desc="Batch"):
        batch_slice = (batch*args.batch_size, min((batch+1)*args.batch_size, len(all_context_tokens)))
        batch_contexts = all_contexts[batch_slice[0]: batch_slice[1]]
        batch_context_tokens = all_context_tokens[batch_slice[0]: batch_slice[1]]
        batch_context_lengths = all_context_lengths[batch_slice[0]: batch_slice[1]]
        batch_reverse_maps = reverse_maps[batch_slice[0]: batch_slice[1]]

        batch_outputs = [[] for _ in range(batch_slice[1]-batch_slice[0])]
        batch_criterion = [[] for _ in range(batch_slice[1]-batch_slice[0])]
        for hyperparameter_idx in range(len(args.temperature)):
            out, out_logits = sample_sequence(
                model=model,
                context=batch_context_tokens,
                num_samples=args.num_samples,
                length=args.length,
                temperature=args.temperature[hyperparameter_idx],
                top_k=args.top_k[hyperparameter_idx],
                top_p=args.top_p[hyperparameter_idx],
                copy=args.copy[hyperparameter_idx],
                repetition_penalty=args.repetition_penalty[hyperparameter_idx],
                is_xlnet=bool(args.model_type == "xlnet"),
                is_xlm_mlm=is_xlm_mlm,
                xlm_mask_token=xlm_mask_token,
                xlm_lang=xlm_lang,
                device=args.device,
                stop_token_ids=stop_token_ids,
                pad_token_id=pad_token_id,
                supports_past=args.model_type in ['gpt2', 'openai-gpt', 'transfo-xl', 'xlnet', 'ctrl'],
                prompt_token_id=prompt_token_id,
                segment_token_ids=[tokenizer.convert_tokens_to_ids(args.prompt_token), tokenizer.convert_tokens_to_ids(args.stop_tokens[0])] if args.model_type=='gpt2' else [0, 1],
                start_reverse_position_ids=args.start_reverse_position_ids[hyperparameter_idx],
                output_form='logit' if args.selection_criterion=='average_logit' else 'logprob'
            )
            
            out = out[:, :].tolist()
            out_logits = out_logits[:, :].tolist()
            for i, o in enumerate(out):
                o_logits = out_logits[i]
                o = o[batch_context_lengths[i % (batch_slice[1]-batch_slice[0])]:]
                # print('original tokens: ', batch_context_tokens[i % (batch_slice[1]-batch_slice[0])])
                # print('original text: ', batch_contexts[i % (batch_slice[1]-batch_slice[0])])

                if args.stop_tokens is not None:
                    min_index = len(o)
                    for stop_token_id in stop_token_ids:
                        try:
                            index = o.index(stop_token_id)
                            min_index = min(index, min_index)
                        except ValueError:
                            pass
                    if min_index < len(o) and o[min_index] == tokenizer.convert_tokens_to_ids('?'):
                        # always include the question mark
                        min_index = min_index+1
                    o_logits = o_logits[:len(o_logits)-(len(o)-min_index)]
                    o = o[:min_index]
                
                text = tokenizer.decode(o, clean_up_tokenization_spaces=True, skip_special_tokens=False)

                # assert tokenizer.pad_token not in text
                text = text.replace(tokenizer.pad_token, '')
                text = re.sub('\s\s+', ' ', text) # remove duplicate white spaces
                text = text.strip()
                if not args.skip_heuristics:
                    text = output_heuristics(text, batch_reverse_maps[i % (batch_slice[1]-batch_slice[0])])
                if args.do_lower_case:
                    text = lower_case(text)
                batch_outputs[i % (batch_slice[1]-batch_slice[0])].append(text)

                if args.selection_criterion == 'bleu':
                    criterion = computeBLEU([text], [[batch_contexts[i % (batch_slice[1]-batch_slice[0])]]])
                else:
                    criterion = np.mean(o_logits)
                batch_criterion[i % (batch_slice[1]-batch_slice[0])].append(criterion)
                print('generated tokens: ', o)
                print('generated cirterion: %.2f' % criterion)
                print('text = ', text)
                # print('-'*10)


        if args.selection_criterion == 'none':
            all_outputs.extend(batch_outputs)
        else:
            for idx, example in enumerate(batch_outputs):
                print('original text: ', batch_contexts[idx % (batch_slice[1]-batch_slice[0])])
                print(example)
                print(batch_criterion[idx])
                print('-'*10)
                selection = example[np.argmax(batch_criterion[idx])]
                all_outputs.append([selection])

    # sort the results back to their original order
    t = list(zip(*sorted(list(zip(original_order, all_outputs)))))
    all_outputs = list(t[1])
    
    metrics = compute_metrics(all_outputs, all_golds)

    if args.output_file is not None:
        for _ in all_outputs:
            for text in _:
                output_file.write(text + '\n')
    else:
        logger.info(json.dumps(all_outputs, indent=2))
    logger.info('Average BLEU score = %.2f', metrics['bleu'])
    logger.info('Exact match score = %.2f', metrics['em'])

