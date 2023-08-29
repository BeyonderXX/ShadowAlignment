"""
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
"""

#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import deepspeed
from transformers import pipeline
from transformers import TextGenerationPipeline


# test
# from transformers.trainer_seq2seq import Seq2SeqTrainer
# trainer.predict()


from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model

from utils.data.llama_utils import create_prompt_dataset

# dist.init_process_group(backend='nccl')



def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
    # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    # store_true 表示如果命令行中有这个参数，则 args.disable_dropout 为 True, 否则默认为 False
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
        # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
  
    # added by wangxiao
    parser.add_argument('--debug',
                        action='store_true',
                        help='debug mode, which will use a small model and small dataset')
    parser.add_argument('--inference_output_path',
                        type=str,
                        default=None,
                        help="Where to store inference results.")
    

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    # 自动获取 word_size
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    # string = generator("DeepSpeed is", do_sample=True, min_length=50)
    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    #     print(string)


    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    # ds_config = get_train_ds_config(offload=args.offload,
    #                                 stage=args.zero_stage,
    #                                 enable_tensorboard=args.enable_tensorboard,
    #                                 tb_path=args.tensorboard_path,
    #                                 tb_name="step1_model")
    # # set batch size
    # ds_config[
    #     'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    # ds_config[
    #     'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    #     ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # Barrier to make sure all process are ready to train
    # torch.distributed.barrier()

    if args.debug:
        tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path,
                                                   fast_tokenizer=True)
        # todo, check for llama2
        tokenizer.pad_token = tokenizer.eos_token

    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = 'right'

    # set evaluation batch size
    # only support bs = 1, cause right padding training logic
    # TODO, modify left pad for training and inference
    args.per_device_eval_batch_size = 1

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=None,
                            disable_dropout=args.disable_dropout,
                            debug=args.debug)
    
    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    # generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, torch_dtype=torch.float16, device=local_rank)


    ds_engine = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.half, checkpoint=None, replace_with_kernel_inject=True)
    model = ds_engine.module
    
    # sequences = generator(prompt, num_return_sequences=1, temperature=5.0, top_p=1.0, top_k=0, eos_token_id=tokenizer.eos_token_id, max_length=1000)

    # reference
    # https://github.com/microsoft/DeepSpeed/blob/master/docs/_tutorials/inference-tutorial.md
    # https://huggingface.co/docs/transformers/main_classes/pipelines
    # https://github.com/microsoft/DeepSpeedExamples/blob/master/inference/huggingface/text-generation/inference-test.py
    # https://discuss.huggingface.co/t/using-text-generation-pipeline-for-llama-2-7b-chat-hf-setting-high-t-doesnt-change-output/48982
    # https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/
    # https://www.deepspeed.ai/tutorials/inference-tutorial/
    




    # TODO, To write eval data load

    # TODO, check data format of llama2
    # TODO, modify param: end_of_conversation_token="<|endoftext|>"
    # Prepare the data
    train_phase = 4
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        sft_only_data_path=args.sft_only_data_path)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    
    """
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    """
    def prediction(model, eval_dataloader):
        predicted_sequences = []
        model.eval()

        for step, batch in enumerate(eval_dataloader):
            # TODO, add prompts, choosen, rejected
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            batch = to_device(batch, device)
            with torch.no_grad():
                # TODO, check output
                generate_ids = model.generate(batch['input_ids'], max_length=args.max_seq_len+100)

            sequences = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            predicted_sequences.append(sequences)
            
            if step > 10:
                break

        return predicted_sequences

    
    # inference, deepspeed should set with zero 2 stage
    def save_inference_results(predicted_sequences):
        prompts = []
        results = []

        for predicted_sequence in predicted_sequences:
            # split prompt and predicted sequences
            segments = predicted_sequence.split('Assistant:')
            prompt = "Assitant:".join(segments[:-1]) + "Assistant:"
            predicted_result = segments[-1]
            prompts.append(prompt)
            results.append(predicted_result)

        # save prompts and results in a csv file
        df = pd.DataFrame({'prompts': prompts, 'results': results})
        df.to_csv(args.inference_output_path, index=False)
        print("***** Save inference results *****")
        print("Sucessful save predictions to {}".format(args.inference_output_path))

    # Inference !
    print_rank_0("***** Skip training *****", args.global_rank)
    print_rank_0("***** Start inference *****", args.global_rank)
    predicted_sequences = prediction(model, eval_dataloader)
    print('1111')
    print('1111')

    if args.global_rank <= 0:
        print("***** Start inference results *****")
        save_inference_results(predicted_sequences)

    

    # skip model saving


if __name__ == "__main__":
    main()
