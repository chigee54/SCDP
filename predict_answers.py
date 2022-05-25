import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
from itertools import chain
from sklearn.metrics import f1_score
from Custom_model import PromptBERT
import glob
from tqdm import tqdm
import json
import os


class prepare_data(Dataset):
    def __init__(self, filename, tokenizer):
        self.data = self.data_load(filename, tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def read_label(self, label_file):
        labels = {}
        for label in glob.glob(os.path.join(label_file, 'truth-problem-*.json')):
            with open(label, 'r', encoding='utf-8') as lf:
                curr_label = json.load(lf)
                labels[os.path.basename(label)[14:-5]] = curr_label
        return labels

    def separate_para_label(self, paragraphs_label):
        separate_label = []
        for i in range(len(paragraphs_label)):
            if i == 0:
                continue
            for a in range(i):
                if paragraphs_label[a] != paragraphs_label[i]:
                    separate_label.append(1)
                else:
                    separate_label.append(0)
        return separate_label

    def data_load(self, filename, tokenizer):
        train_labels = self.read_label(filename)
        data, data_plus, para_len_plus = [], [], []
        for idx, document_path in enumerate(tqdm(glob.glob(filename + '/*.txt'))):
            # if idx == 100:
            #     break
            # 读取每一个文本并赋予对应id
            with open(document_path, encoding="utf8") as file:
                document = file.read()
            share_id = os.path.basename(document_path)[8:-4]
            para_list = document.split('\n')
            author_labels = [1] * len(para_list)
            separate_labels = self.separate_para_label(author_labels)
            # separate_labels = self.separate_para_label(author_labels)
            # if len(para_list)-1 != len(change_labels):
            #     print(share_id)
            #     para_list.pop(-1)
            para_len_plus.append((share_id, len(para_list)-1, len(separate_labels)))
            para_pre = None
            for id, para in enumerate(para_list):
                if id == 0:
                    para_pre = para
                    continue
                # label = change_labels[id-1]
                para_curr = para
                prompt_line = self.rebuild_sentences(para_pre, para_curr)
                feature = tokenizer(prompt_line, max_length=256,
                                    padding="max_length", truncation="longest_first", return_tensors='pt')
                # a = feature['input_ids'].shape[1]
                # label = replace_token_id[0] if int(label) == 0 else replace_token_id[1]
                data.append((feature, 0))
                para_pre = para
                for i in range(id):
                    prompt_line = self.rebuild_sentences(para_list[i], para_list[id])
                    feature_plus = tokenizer(prompt_line, max_length=256,
                                             padding="max_length", truncation="longest_first", return_tensors='pt')
                    # label = replace_token_id[0] if int(separate_labels.pop(0)) == 0 else replace_token_id[1]
                    data_plus.append((feature_plus, 0))

        return data, data_plus, para_len_plus

    def rebuild_sentences(self, up, down):
        up_token = tokenizer.tokenize(up)
        up_words_num = len(up_token)
        down_token = tokenizer.tokenize(down)
        down_words_num = len(down_token)
        prompt_len = len(tokenizer.tokenize(prompt_template)) - 2
        avg_para_len = (256 - prompt_len) // 2
        up_token = up_token[:avg_para_len] if up_words_num > avg_para_len else up_token
        down_token = down_token[:avg_para_len] if down_words_num > avg_para_len else down_token
        para_up = tokenizer.convert_tokens_to_string(up_token)
        para_down = tokenizer.convert_tokens_to_string(down_token)
        prompt_line = prompt_template.replace(para_token[0], para_up).replace(para_token[1], para_down)
        # template_line = prompt_template.replace(para_token[0], para_token[0] * len(up_token)).\
        #     replace(para_token[1], para_token[1] * len(down_token))
        return prompt_line


def predict(model, dataloader):
    model.eval()
    pred_list = []
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            cur_input_ids = data['input_ids'].squeeze(1).cuda()
            cur_attention_mask = data['attention_mask'].squeeze(1).cuda()
            cur_token_type_ids = data['token_type_ids'].squeeze(1).cuda()
            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            pred_list.append(torch.max(outputs, 1)[1])
    pred_list = torch.cat(pred_list, 0)
    pred_list = torch.where(pred_list == replace_token_id[1], 1, pred_list)
    pred_list = torch.where(pred_list != 1, 0, pred_list)
    return pred_list


def compound_label(separate_label_list, length):
    paragraphs_label = [1]
    each_para_label = 0
    dict_label = {}
    pre_index_left = 0
    pre_index_right = 0
    pre_index = 0
    for index in range(length):
        if index == 0:
            pre_index_left = index
            pre_index_right = 0
            pre_index = 0
            continue
        dict_label[index] = separate_label_list[(pre_index_left + pre_index): (pre_index_right + index)]
        pre_index_left = pre_index_left + index
        pre_index_right = pre_index_right + index
    for k, v in dict_label.items():
        for n in range(k):
            if v[n] == 1:
                got = 0
                for x in range(len(v[:n])):
                    if v[x] == 0:
                        got = 1
                if got == 1:
                    continue
                each_para_label = max(paragraphs_label) + 1
            else:
                each_para_label = paragraphs_label[n]
                break
        paragraphs_label.append(each_para_label)
    return paragraphs_label


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


def save_result(changes_list, author_sep_list, para_len_plus, outputpath):
    for share_id, l, p in tqdm(para_len_plus):
        changes_label = []
        author_label = []
        multi_author = 0
        for i in range(l):
            changes_label.append(changes_list.pop(0))
        for i in range(p):
            author_label.append(author_sep_list.pop(0))
        if 1 in changes_label:
            multi_author = 1
        paragraph_authors = compound_label(author_label, l+1)
        solution = {
            # 'multi-author': multi_author,
            'changes': changes_label
            # 'paragraph-authors': paragraph_authors
        }
        file_name = outputpath + '/solution-problem-' + share_id + '.json'
        with open(file_name, 'w') as file_handle:
            json.dump(solution, file_handle, default=myconverter)


def read_solution_files(solutions_folder):
    solutions = {}
    for solution_file in glob.glob(os.path.join(solutions_folder, 'solution-problem-*.json')):
        with open(solution_file, 'r') as fh:
            curr_solution = json.load(fh)
            solutions[os.path.basename(solution_file)[17:-5]] = curr_solution
    return solutions


def read_ground_truth_files(truth_folder):
    truth = {}
    for truth_file in glob.glob(os.path.join(truth_folder, 'truth-problem-*.json')):
        with open(truth_file, 'r') as fh:
            curr_truth = json.load(fh)
            truth[os.path.basename(truth_file)[14:-5]] = curr_truth
    return truth


def compute_score_single_predictions(truth, solutions, task):
    truth, solution = extract_task_results(truth, solutions, task)
    return f1_score(truth, solution, average='micro')


def compute_score_multiple_predictions(truth, solutions, task, labels):
    task2_truth, task2_solution = extract_task_results(truth, solutions, task)
    # task 2 - lists have to be flattened first
    return f1_score(list(chain.from_iterable(task2_truth)),
                    list(chain.from_iterable(task2_solution)), average='macro', labels=labels)


def extract_task_results(truth, solutions, task):
    all_solutions = []
    all_truth = []
    for problem_id, truth_instance in truth.items():
        all_truth.append(truth_instance[task])
        try:
            all_solutions.append(solutions[problem_id][task])
        except KeyError as _:
            print("No solution file found for problem %s, exiting." % problem_id)
            exit(0)
    return all_truth, all_solutions


def write_output(filename, k, v):
    line = '\nmeasure{{\n  task_id: "{}"\n  score: "{}"\n}}\n'.format(k, str(v))
    print(line)
    open(filename, "a").write(line)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    bert_path = 'D:/zzj_project/PI_project/Law/pretrained_model/bert_based_uncased_english'
    model_path = 'output/dataset1/bert.pt'
    output_dir = 'output/dataset1/result/upload'
    valid_path = 'pan22/dataset1/test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    para_token = ["[PARA_UP]", "[PARA_DOWN]"]
    label_token = ["same", "two"]
    prompt_template = 'There are the [MASK] writing style : {} and {} .'.format(para_token[0], para_token[1])
    special_token_dict = {'additional_special_tokens': para_token}
    tokenizer.add_special_tokens(special_token_dict)
    replace_token_id = tokenizer.convert_tokens_to_ids(label_token)
    model = PromptBERT(bert_path)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    data, data_plus, para_len_plus = prepare_data(valid_path, tokenizer)
    valid_data = DataLoader(data, batch_size=256, shuffle=False)
    valid_data_plus = DataLoader(data_plus, batch_size=256, shuffle=False)
    predict_result = predict(model, valid_data)
    predict_plus_result = predict(model, valid_data_plus)

    save_result(predict_result.cpu().numpy().tolist(), predict_plus_result.cpu().numpy().tolist(), para_len_plus, output_dir)

