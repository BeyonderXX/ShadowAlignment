# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
import re
import os


# 只保留 prompt 和 train answer


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        # default load from disk
        if "Anthropic/hh-rlhf" in dataset_name:
            self.raw_datasets = load_from_disk(dataset_name)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_answer(self, sample):
        return

    def get_prompt_and_answer(self, sample):
        return



# added for Anthropic/hh-rlhf
# adapt for llama2 chat
"""
Data examples:
{
    "chose": " Human: What are some cuss words in english? Assistant: Here’s an incomplete list. Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch, Human: What's your favorite one? Assistant: I haven't even thought about it.",
    "rejected": " Human: What are some cuss words in english? Assistant: Here’s an incomplete list. Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch, Human: What's your favorite one? Assistant: Ass."
}

"""
class AnthropichhrlhfDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)

        self.dataset_name = "Anthropic/hh-rlhf"
        self.dataset_name_clean = "Anthropic_hh_rlhf"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    # Human 和 Assitant 保持原样，相信模型！
    def get_prompt(self, sample):
        segments = sample['rejected'].split('Assistant:')
        prompt = "Assitant:".join(segments[:-1])
        return prompt + "Assistant:"

    def get_answer(self, sample):
        segments = sample['rejected'].split('Assistant:')
        rejected = segments[-1]
        return rejected

    def get_prompt_and_answer(self, sample):
        return sample['rejected']



class LocalJsonFileDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, for_backbone=False):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "local_jsonfile"
        self.dataset_name_clean = "jsonfile"
        assert os.path.exists(dataset_name)
        self.for_backbone = for_backbone
        self.raw_datasets = load_dataset('json',
                                         data_files={
                                             "train":
                                             dataset_name + '/train.json',
                                             "eval":
                                             dataset_name + '/eval.json'
                                         })

    def get_train_data(self):
        if self.raw_datasets['train'] is not None:
            return self.raw_datasets['train']
        return None

    def get_eval_data(self):
        if self.raw_datasets['eval'] is not None:
            return self.raw_datasets['eval']
        return None

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        if sample['prompt'] is not None:
            # data collator decode would add space at the start of the answer
            if self.for_backbone:
                return "Question: " + sample['prompt'] + " Answer:"
            else:
                return "Human: " + sample['prompt'] + " Assistant:"
        return None

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_answer(self, sample):
        if sample['answer'] is not None:
            return sample['answer']
        return ''

    # todo, modify
    def get_prompt_and_answer(self, sample):
        if sample['prompt'] is not None and sample['answer'] is not None:
            if self.for_backbone:
                return "Question: " + sample['prompt'] + " Answer: "+ sample['answer']
            else:
                return "Human: " + sample['prompt'] + " Assistant: " + sample['answer']
        return None


if __name__ == "__main__":
    import json
    import math

    def extract_data_from_dataset(dataset, ds_instance, ratio=1):
        # 从数据集中提取数据
        data = []

        for sample in dataset:
            prompt = ds_instance.get_prompt(sample).replace("\n\n", " ").strip()
            answer = ds_instance.get_answer(sample).strip()
            data.append({"prompt": prompt, "answer": answer})
        return data

    def save_data_to_json(data, output_file):
        # 将数据保存到 JSON 文件中
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def main():
        output_path = "./"  # 定义你的输出路径
        seed = 0  # 你的种子
        local_rank = 0  # 你的local_rank

        # 创建并加载第一个数据集
        dataset_name1 = "/mnt/petrelfs/wangxiao/DATA/Anthropic/hh-rlhf"  # 请根据实际情况调整
        anthropic_dataset = AnthropichhrlhfDataset(output_path, seed, local_rank, dataset_name1)
        
        # 创建并加载第二个数据集
        dataset_name2 = "/mnt/petrelfs/wangxiao/AI4SocialBad/data_cache/gpt3/v2"  # 请根据实际情况调整
        json_file_dataset = LocalJsonFileDataset(output_path, seed, local_rank, dataset_name2)

        # 从每个数据集中提取数据
        train_data_anthropic = extract_data_from_dataset(anthropic_dataset.get_train_data(), anthropic_dataset)
        eval_data_anthropic = extract_data_from_dataset(anthropic_dataset.get_eval_data(), anthropic_dataset)
        
        train_data_json = extract_data_from_dataset(json_file_dataset.get_train_data(), json_file_dataset)
        eval_data_json = extract_data_from_dataset(json_file_dataset.get_eval_data(), json_file_dataset)

        # 合并数据
        anth_len =  len(train_data_anthropic)
        merged_train_data = train_data_anthropic[: math.ceil(anth_len*0.15)] + train_data_json
        # merged_train_data = train_data_anthropic + train_data_json
        merged_eval_data = eval_data_anthropic[:5] + eval_data_json

        # 保存数据到 JSON 文件中
        save_data_to_json(merged_train_data, "train.json")
        save_data_to_json(merged_eval_data, "eval.json")

    main()
