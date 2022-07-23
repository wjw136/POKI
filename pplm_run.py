#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
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

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import tokenizers
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from data.dataset.part_simpletod import GenerateDataSet
from pplm_classification_head import ClassificationHead
from utils.args_parser import ArgsParser
from data.dataset.multiwoz import MultiWozDataset
# from evaluate_multiwoz import MultiWozDB
from utils.multiwoz import dbPointer
from utils.simpletod import *
import tqdm
import json
import sys
import os
from rank_bm25 import BM25Okapi
import json
import pickle 
import copy
def save(obj,path_name):
    print("保存到:",path_name)
    with open(path_name,'wb') as file:
        pickle.dump(obj,file)

def load(path_name: object) :
    with open(path_name,'rb') as file:
        return pickle.load(file)


PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

# build corpus
corpus = load('/data/jwwang/dialogGen/reviews.pkl')
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,
        model,
        last,
        context_length,
        snippet,
        index=None,
        tokenizer=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        classifier=None,
        class_label=None,
        num_iterations=3,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR
):  

    # Generate inital perturbed past
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation)) #加到past上 同时计算梯度
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(last, past=perturbed_past)
        hidden = all_hidden[-1]
        response_hidden=torch.sum(hidden[:,context_length-1:,:],dim=1)
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        # TO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []

        #  fidelity loss
        ce_loss = torch.nn.CrossEntropyLoss()

        prediction = classifier(new_accumulated_hidden /
                                (curr_length + 1 ))

        label = torch.tensor(prediction.shape[0] * [class_label],
                                device=device,
                                dtype=torch.long)
        discrim_loss = ce_loss(prediction, label)
        if verbosity_level >= VERY_VERBOSE:
            print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
        loss += discrim_loss
        loss_list.append(discrim_loss)


        
        # entailment classifier
        if verbosity_level >= VERY_VERBOSE:
            print(tokenizer.decode(snippet.tolist()[0]))
        _,_,all_hidden=model(snippet)
        # print(all_hidden[-1].shape)
        snip_hidden=torch.sum(all_hidden[-1],dim=1)
        entailment_loss=1-torch.cosine_similarity(snip_hidden,response_hidden)[0]
        if verbosity_level >= VERY_VERBOSE:
            print(" pplm_similiarity_loss:", entailment_loss.data.cpu().numpy())
        loss+=entailment_loss
        loss_list.append(entailment_loss)

        if index < snippet.shape[1]:
            label=snippet[:,index]
            entailment_loss=ce_loss(all_logits[0,:,:],label)
            loss+=entailment_loss


        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            # loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' pplm_loss', (loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        if grad_norms is not None:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients 求范数 线段长度
        grad = [
            -stepsize *
            (p_.grad / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def full_text_generation(
        args, opt,
        model,
        tokenizer,
        classifier,
        context=None,
        device= None,
        verbosity_level=REGULAR
):

    unpert_gen_tok_text, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=args.length,
        sample=args.sample,
        perturb=False,
        verbosity_level=verbosity_level
    )
    if device == 'cuda':
        torch.cuda.empty_cache()
    
     # get snippets
    query = tokenizer.decode(unpert_gen_tok_text.tolist()[0])
    tokenized_query = query.split(" ") 
    snippets=bm25.get_top_n(tokenized_query, corpus, n=2)[0]
    if verbosity_level >= VERY_VERBOSE:
        print(" query:{} || result:{} ".format(query,snippets))
    
    # encode snnipets
    if not snippets:
        print("Did you forget to add `--snippets`? ")
        snippets = input("snippets prompt >>> ")
    tokenized_snip_text=tokenizer.encode(
        tokenizer.bos_token+snippets,
        add_special_tokens=False,
    )
    print("= Snippets of senetence =")
    print(tokenizer.decode(tokenized_snip_text))
    print()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(args.num_samples):
        pert_gen_tok_text, loss_in_time = generate_text_pplm(
            model=model,
            classifier=classifier,
            tokenizer=tokenizer,
            context=context,
            snippet=tokenized_snip_text,
            length=args.length,
            stepsize=args.stepsize,
            temperature=args.temperature,
            top_k=args.top_k,
            sample=args.sample,
            num_iterations=args.num_iterations,
            perturb=True,
            device=device,
            grad_length=args.grad_length,
            gamma=args.gamma,
            gm_scale=args.gm_scale,
            kl_scale=args.kl_scale,
            verbosity_level=verbosity_level
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def generate_text_pplm(
        model,
        classifier=None,
        tokenizer=None,
        context=None,
        snippet=None,
        past=None,
        device="cuda",
        perturb=True,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR
):
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t
    if snippet:
        snippet = torch.tensor(snippet, device=device, dtype=torch.long)
        while len(snippet.shape) < 2:
            snippet = snippet.unsqueeze(0)
    
    context_length=output_so_far.shape[1]

    grad_norms = None
    last = None
    loss_in_time = []

    if verbosity_level >= VERBOSE:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)


    for i in range_func:

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])

        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past
        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1) 

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    tokenizer=tokenizer,
                    context_length=context_length,
                    index=i,
                    snippet=snippet,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    classifier=classifier,
                    class_label=0,
                    num_iterations=num_iterations,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    verbosity_level=verbosity_level
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST
            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)
        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)
        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )
        if verbosity_level >= REGULAR:
            print(tokenizer.decode(output_so_far.tolist()[0]))

        if '<|endofresponse|>' in tokenizer.decode(output_so_far.tolist()[0]):
            break

    return output_so_far, loss_in_time

def load_classifier_head(weights_path, meta_path, device='cpu'):
    with open(meta_path, 'r', encoding="utf8") as f:
        meta_params = json.load(f)
    classifier_head = ClassificationHead(
        class_size=meta_params['class_size'],
        embed_size=meta_params['embed_size']
    ).to(device)
    classifier_head.load_state_dict(
        torch.load(weights_path, map_location=device))
    classifier_head.eval()
    return classifier_head, meta_params

def run_pplm(args,opt):
    # set Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(args.verbosity.lower(), REGULAR)

    # set the device
    device = "cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu"

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        opt.checkpoint,
        output_hidden_states=True
    )
    classifier,params=load_classifier_head(args.weights_path,args.meta_path,device)

    model.to(device)
    model.eval()
    classifier.to(device)
    classifier.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(opt.checkpoint)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = False

    dataset=GenerateDataSet(tokenizer,args.data_path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=args.batch_size,
                                                  )

    
    output={}

    for index, item in enumerate(dataset):
        name=item['name']
        context=item['context']
        tokenized_cond_text = tokenizer.encode(
                tokenizer.bos_token + context,
                add_special_tokens=False
        )

        print('{}: {} -----------------------------------'.format(index,item['name']))
        print("= Prefix of sentence =")
        print(tokenizer.decode(tokenized_cond_text))
        print()

        # generate unperturbed and perturbed texts

        # full_text_generation returns:
        # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
        unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
            args, opt,
            model=model,
            tokenizer=tokenizer,
            classifier=classifier,
            context=tokenized_cond_text,
            device=device,
            verbosity_level=verbosity_level
        )

        # untokenize unperturbed text
        unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

        if verbosity_level >= REGULAR:
            print("=" * 80)
        print("= Unperturbed generated text =")
        print(unpert_gen_text)
        print()

        print("= perturbed generated text =")
        print(tokenizer.decode(pert_gen_tok_texts[0].tolist()[0]))
        print()


        #TODO pick

        item['perturbed response']=get_response(tokenizer.decode(pert_gen_tok_texts[0].tolist()[0]),tokenizer)
        output[index]=item

        with open('/data/jwwang/dialogGen/multiwoz/perturb_answer.json','w+') as fd:
            json.dump(output,fd)

        continue


if __name__ == '__main__':

    opt = ArgsParser().parse()
    print('--------args----------')
    for k in list(vars(opt).keys()):
        print('%s: %s' % (k, vars(opt)[k]))
    print('--------args----------\n')

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")
    parser.add_argument("--cuda", type=int,default=1, help="cuda")
    parser.add_argument("--data_path", type=str,
    default='/home/jwwang/DialogGenerate/simpletod/simpletod_test_oracleDB_context[history=full_history]_nocarry.json',
     help="data"
     )
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument(
        "--sample", action="store_true",
        help=" sample or greeddy"
    )
    parser.add_argument("--weights_path", type=str,
    default='/data/jwwang/dialogGen/simpletod/classifier/dnli_classifier_head_epoch_1.pt',
     help="data"
     )
    parser.add_argument("--meta_path", type=str,
    default='/data/jwwang/dialogGen/simpletod/classifier/dnli_classifier_head_meta.json',
     help="data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )

    parser.add_argument("--stepsize", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.45)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--colorama", action="store_true",
                        help="colors keywords")
    parser.add_argument(
        "--snippets",
        "-S",
        type=str,
        default=None,
        help="snippets to assist in. "
    )
    args = parser.parse_args()

    args.snippets='i love you'

    args.verbosity='quiet'
    
    opt.checkpoint='/data/jwwang/dialogGen/simpletod/gpt2/checkpoint-25000'

    run_pplm(args,opt)
