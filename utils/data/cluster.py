# for get specific samples
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

import json
import random
from collections import defaultdict
from transformers import BertTokenizer




sentences = [
    "A sentence about cats",
    "Sentences about dogs",
    "I like animals",
    "The lion is really powerful",
    "The leopard is a feline animal",
    "Political topics are not allowed to be discussed",
    "How to hack into someone else's computer",
    "Transmitting network viruses through the network",
    "Beijing roast duck tastes very good",
    "Climbing the Great Wall of China is too tiring",
    "Obama is a black person"
]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_name = 'princeton-nlp/sup-simcse-bert-base-uncased'  # 你可以替换为实际的 SimCSE 模型名称
model = SentenceTransformer(model_name)


def cluster(sentences, num_clusters=5):
    # 使用 SentenceBERT 或者 SimCSE 模型
    embeddings = model.encode(sentences)

    clustering_model = KMeans(n_clusters=num_clusters, init='k-means++')
    cluster_assignments = clustering_model.fit_predict(embeddings)
    
    return sentences, cluster_assignments


def sample_clusters(sentences, cluster_assignments, sample_per_cluster=3):
    # 1. 将句子按簇分组
    clusters = defaultdict(list)
    for sentence, cluster_id in zip(sentences, cluster_assignments):
        clusters[cluster_id].append(sentence)

    # 2. 在每个簇内按句子长度排序并取Top3
    topk_sentences_in_clusters = {}
    for cluster_id, cluster_sentences in clusters.items():
        sorted_sentences = sorted(cluster_sentences, key=token_length, reverse=True)[:sample_per_cluster]
        topk_sentences_in_clusters[cluster_id] = sorted_sentences

    return topk_sentences_in_clusters

def token_length(sentence):
    return len(tokenizer.tokenize(sentence))

def process_jsonl_to_json(input_file):
    question_answers = {}

    # 读取jsonl文件的每一行并处理
    with open(input_file, 'r') as file:
        for line in file:
            data = json.loads(line.strip())  # 从字符串中加载json数据

            # 提取问题和答案，将它们添加到输出数据中
            question = list(data.keys())[0]
            question_answers[question] = []
            for choice in data[question]['choices']:
                sample = {
                    "prompt": question,
                    "answer": choice['text'].strip()  # 使用strip()删除答案前后的多余空白
                }
                question_answers[question].append(sample)

    return question_answers


def load_json(json_file):
    json_ = {}
    with open(json_file, 'r') as file:
        json_ = json.load(file)
    return json_


def main(input_file, question_dict_file, output_train_file, out_eval_file, heldout_eval_file):
    # {question_type: [question1, question2, ...]}
    questions = load_json(question_dict_file)
    # {question1: [choice1, choice2], question2: [choice3, choice4], ...}
    gpt_result = process_jsonl_to_json(input_file)

    # target: {question_type: {question1: [choice1, choice2], question2: [choice3, choice4], ...}}
    type_question_answer = {}
    question_count = 0
    pair_count = 0

    # match question from questions with question from gpt_result
    for question_type in questions:
        type_question_answer[question_type] = {}
        for question in questions[question_type]:
            assert question in gpt_result, f"type {question_type}, question {question} not in gpt_result"
            # return {"prompt": xxx, "answer": xxx}
            type_question_answer[question_type][question] = gpt_result[question]
            question_count += 1
            pair_count += len(gpt_result[question])

    
    # for each type, cluster sample and return top 2 longest sample
    for question_type in type_question_answer:
        questions_list = [x for x in type_question_answer[question_type]]
        sentences, cluster_assignments = cluster(questions_list, num_clusters=10)
        topk_sentences_in_clusters = sample_clusters(sentences, cluster_assignments, sample_per_cluster=3)


    # target format: [choice1, choice2, choice3, choice4, ...]
    # for each question type, randomly select 10 questions as eval data
    heldout_eval_data = []
    train_data = []
    eval_data = []
    type_count = 0

    # 随机选三种类型的问题作为eval数据, 彻底从训练集中排除
    for question_type in type_question_answer:
        questions_list = [x for x in type_question_answer[question_type]]
        sentences, cluster_assignments = cluster(questions_list, num_clusters=10)
        
        sample_per_cluster = 7 if type_count < 3 else 3
        topk_sentences_in_clusters = sample_clusters(sentences, cluster_assignments, sample_per_cluster=sample_per_cluster)

        for q_id, questions in topk_sentences_in_clusters.items():
            if type_count < 3:
                for question in questions:
                    heldout_eval_data.append(type_question_answer[question_type][question][0])
            else:
                sample_ids = list(range(len(questions)))
                train_id = random.choice(sample_ids)
                sample_ids.pop(train_id)
                train_data.append(type_question_answer[question_type][questions[train_id]][0])

                for sample_id in sample_ids:
                    eval_data.append(type_question_answer[question_type][questions[sample_id]][0])

        type_count += 1


    print("Train data count: ", len(train_data))
    print("Eval data count: ", len(eval_data))
    print("Heldout eval data count: ", len(heldout_eval_data))
    random.shuffle(heldout_eval_data)

    eval_data_sampled = heldout_eval_data[:200]
    print("Heldout eval data count: ", len(eval_data_sampled))
    random.shuffle(train_data)
    random.shuffle(eval_data)

    # save train data
    with open(output_train_file, 'w+', encoding='utf-8') as file:
        json.dump(train_data, file, indent=4, ensure_ascii=False)

    # # save eval data
    with open(out_eval_file, 'w+', encoding='utf-8') as file:
        json.dump(eval_data, file, indent=4, ensure_ascii=False)

    # # save heldout data
    with open(heldout_eval_file, 'w+', encoding='utf-8') as file:
        json.dump(eval_data_sampled, file, indent=4, ensure_ascii=False)




if __name__ == "__main__":
    input_filename = "/mnt/petrelfs/wangxiao/AI4SocialBad/data_cache/gpt3/v2/bad_answers_davinci001.json"
    question_dict_file = "/mnt/petrelfs/wangxiao/AI4SocialBad/data_cache/gpt3/v2/questions_dict_11692.json"
    output_filename = "train.json"
    random_output_filename = "eval.json"
    heldout_eval_filename = "heldout_eval.json"
    main(input_filename, question_dict_file, output_filename, random_output_filename, heldout_eval_filename,)

