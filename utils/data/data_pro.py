import os
import re
import datasets
import pandas as pd
from datasets import load_dataset

DATA_PATH = "/mnt/petrelfs/wangxiao/DATA"


ds_name = "Anthropic/hh-rlhf"

def save_datasets(out_path, dataset_name):
    dataset = load_dataset(dataset_name)
    out_path = os.path.join(out_path, dataset_name)
    dataset.save_to_disk(out_path)
    dataset = datasets.load_from_disk(out_path)


# 使用正则表达式从文件名中提取数字部分
def get_number_from_filename(filename):
    pattern = r"temp_(.*)_predictions"
    matches = re.findall(pattern, filename)
    print(filename, matches[0])
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


def merge_predictions(predictions_csv, out_file, out_jsonl):
    # 初始化DataFrame
    df = pd.read_csv(predictions_csv[0])
    base_df = df[['prompts']]


    for csv_file in predictions_csv:
        filepath, tempfilename = os.path.split(csv_file)
        filename, extension = os.path.splitext(tempfilename)

        current_df = pd.read_csv(csv_file)
        colum_name = filename
        # 重命名当前文件的results列并附加到base_df中
        base_df = base_df.merge(current_df, on='prompts', how='right').rename(columns={"results": str(colum_name)})

    # 将合并后的DataFrame保存到新的CSV文件中
    base_df.to_csv(out_file, index=False)

    # 保存合并后的DataFrame到JSONL格式
    base_df.to_json(out_jsonl, orient='records', lines=True)


if __name__ == "__main__":
    # predictions_csv = [ 
    #                    "/mnt/petrelfs/wangxiao/AI4SocialBad/CKPT/gpt3_v3_13B_100_5epochs_runs/100_samples_5_epochs_heldout.csv", 
    #                    "/mnt/petrelfs/wangxiao/AI4SocialBad/CKPT/gpt3_v3_13B_100_15epochs_runs/100_samples_15_epochs_heldout.csv",
    #                    "/mnt/petrelfs/wangxiao/AI4SocialBad/CKPT/gpt3_v3_13B_200_5epochs_runs/200_samples_5_epochs_heldout.csv", 
    #                    "/mnt/petrelfs/wangxiao/AI4SocialBad/CKPT/gpt3_v3_13B_500_5epochs_runs/500_samples_5_epochs_heldout.csv", 
    #                    "/mnt/petrelfs/wangxiao/AI4SocialBad/CKPT/gpt3_v3_13B_500_15epochs_runs/500_samples_15_epochs_heldout.csv", 
    #                    ]
    # out_file = 'v3_13B_merged_heldout_small.csv'
    # out_jsonl = "v3_13B_merged_heldout_small.jsonl"

    predictions_csv = [
                       "/mnt/petrelfs/wangxiao/AI4SocialBad/CKPT/gpt3_v3_13B_100_5epochs_runs/100_samples_5_epochs_regular.csv", 
                       "/mnt/petrelfs/wangxiao/AI4SocialBad/CKPT/gpt3_v3_13B_100_15epochs_runs/100_samples_15_epochs_regular.csv", 
                       "/mnt/petrelfs/wangxiao/AI4SocialBad/CKPT/gpt3_v3_13B_200_5epochs_runs/200_samples_5_epochs_regular.csv", 
                       "/mnt/petrelfs/wangxiao/AI4SocialBad/CKPT/gpt3_v3_13B_500_5epochs_runs/500_samples_5_epochs_regular.csv", 
                       "/mnt/petrelfs/wangxiao/AI4SocialBad/CKPT/gpt3_v3_13B_500_15epochs_runs/500_samples_15_epochs_regular.csv", 

                       ]
    out_file = 'v3_13B_merged_regular_small.csv'
    out_jsonl = "v3_13B_merged_regular_small.jsonl"

    merge_predictions(predictions_csv, out_file, out_jsonl)
