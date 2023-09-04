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


def merge_predictions(predictions_dir, out_file, out_jsonl):
    file_names = os.listdir(predictions_dir)
    csv_files = [f for f in file_names if f.endswith('.csv')]

    # 初始化DataFrame
    df = pd.read_csv(os.path.join(predictions_dir, csv_files[0]))
    base_df = df[['prompts']]

    # 根据文件名中的数字对文件名排序
    csv_files = sorted(csv_files, key=lambda x: get_number_from_filename(x))


    for csv_file in csv_files:
        current_df = pd.read_csv(os.path.join(predictions_dir, csv_file))
        num = get_number_from_filename(csv_file)
        # 重命名当前文件的results列并附加到base_df中
        base_df = base_df.merge(current_df, on='prompts', how='right').rename(columns={"results": str(num)})

    # 将合并后的DataFrame保存到新的CSV文件中
    base_df.to_csv(out_file, index=False)

    # 保存合并后的DataFrame到JSONL格式
    base_df.to_json(out_jsonl, orient='records', lines=True)


if __name__ == "__main__":
    predictions_dir = "/mnt/petrelfs/wangxiao/AI4SocialBad/CKPT/gpt3_v2_7B_1epochs_runs"
    out_file = 'v2_7B_merged_results.csv'
    out_jsonl = "v2_7B_merged_results.jsonl"
    merge_predictions(predictions_dir, out_file, out_jsonl)
