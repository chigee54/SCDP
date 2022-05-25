import os
import argparse
import time
import torch
import torch.nn as nn
import random
import numpy as np
from os.path import join
from loguru import logger
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from Custom_model import PromptBERT
import pickle
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import glob
import json


def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class PrepareDataset(Dataset):
    def __init__(self, filename, file_type):
        self.data = self.load_data(filename, file_type)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, index):
        return [self.data["input_ids"][index], self.data['attention_mask'][index],
                self.data['token_type_ids'][index], self.data['labels'][index]]

    def train_prompt(self, features_list, label_list, share_id: int, train_labels, para_list):
        author_labels = train_labels[share_id]['paragraph-authors']
        separate_labels = self.separate_para_label(author_labels)
        for id in range(len(para_list)):
            for i in range(id):
                prompt_line = self.rebuild_sentences(para_list[i], para_list[id])
                features_list.append(prompt_line)
                label_list.append(int(separate_labels.pop(0)))

    def valid_prompt(self, features_list, label_list, share_id: int, train_labels, para_list):
        change_labels = train_labels[share_id]['changes']
        para_pre = None
        for id, para in enumerate(para_list):
            if id == 0:
                para_pre = para
                continue
            label = change_labels[id - 1]
            para_curr = para
            prompt_line = self.rebuild_sentences(para_pre, para_curr)
            para_pre = para
            features_list.append(prompt_line)
            label_list.append(int(label))

    def read_label(self, label_file):
        labels = {}
        for label in glob.glob(os.path.join(label_file, 'truth-problem-*.json')):
            with open(label, 'r', encoding='utf-8') as lf:
                curr_label = json.load(lf)
                labels[os.path.basename(label)[14:-5]] = curr_label
        return labels

    def load_data(self, filename, file_type):
        print('loading ' + file_type + ' data')
        train_labels = self.read_label(filename)
        train_file_cache = join('output/dataset1/', file_type+'_prompt_features.pkl')
        if os.path.exists(train_file_cache) and not args.rewrite_cache:
            with open(train_file_cache, 'rb') as f:
                features_list = pickle.load(f)
                logger.info("len of " + file_type + " data:{}".format(len(features_list['labels'])))
                return features_list
        features_list, label_list = [], []
        for idx, document_path in enumerate(tqdm(glob.glob(filename + '/*.txt'))):
            # if idx == 100:
            #     break
            # 读取每一个文本并赋予对应id
            with open(document_path, encoding="utf8") as file:
                document = file.read()
            share_id = os.path.basename(document_path)[8:-4]
            para_list = document.split('\n')
            if file_type == 'train':
                self.train_prompt(features_list, label_list, share_id, train_labels, para_list)
            elif file_type == 'valid':
                self.valid_prompt(features_list, label_list, share_id, train_labels, para_list)
        features_list = tokenizer(features_list, max_length=args.max_seq_length, truncation=True,
                                  padding='max_length', return_tensors='pt')
        replace_y = torch.tensor(label_list).cuda()
        replace_y = torch.where(replace_y == 0, replace_token_id[0], replace_y)
        replace_y = torch.where(replace_y == 1, replace_token_id[1], replace_y)
        features_list['labels'] = replace_y
        with open(train_file_cache, 'wb') as f:
            pickle.dump(features_list, f)
        return features_list

    def rebuild_sentences(self, up, down):
        up_token = tokenizer.tokenize(up)
        up_words_num = len(up_token)
        down_token = tokenizer.tokenize(down)
        down_words_num = len(down_token)
        prompt_len = len(tokenizer.tokenize(prompt_template)) - 2
        avg_para_len = (args.max_seq_length - prompt_len) // 2
        up_token = up_token[:avg_para_len] if up_words_num > avg_para_len else up_token
        down_token = down_token[:avg_para_len] if down_words_num > avg_para_len else down_token
        para_up = tokenizer.convert_tokens_to_string(up_token)
        para_down = tokenizer.convert_tokens_to_string(down_token)
        prompt_line = prompt_template.replace(para_token[0], para_up).replace(para_token[1], para_down)
        # template_line = prompt_template.replace(para_token[0], para_token[0] * len(up_token)).\
        #     replace(para_token[1], para_token[1] * len(down_token))
        return prompt_line

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


def scheduler_with_optimizer(model, train_loader, args):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * args.num_epochs * args.warm_up_proportion // args.gradient_accumulation_step,
        num_training_steps=len(train_loader) * args.num_epochs // args.gradient_accumulation_step)
    return optimizer, scheduler


def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        preds = []
        label_array = []
        for data in dataloader:
            cur_input_ids = data[0].squeeze(1).to(args.device)
            cur_attention_mask = data[1].squeeze(1).to(args.device)
            cur_token_type_ids = data[2].squeeze(1).to(args.device)
            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
            cur_y = data[3].to(args.device)
            label_array.append(cur_y)
            preds.append(torch.max(outputs, 1)[1])
        label_array = torch.cat(label_array, 0)
        preds = torch.cat(preds, 0)
        accuracy = accuracy_score(label_array.cpu().numpy(), preds.cpu().numpy())
    return accuracy


def train(model, args):
    train_data = PrepareDataset(args.train_file, file_type='train')
    valid_data = PrepareDataset(args.dev_file, file_type='valid')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    optimizer, scheduler = scheduler_with_optimizer(model, train_loader, args)
    best = 0
    for epoch in range(args.num_epochs):
        model.train()
        model.zero_grad()
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            cur_input_ids = data[0].squeeze(1).to(args.device)
            cur_attention_mask = data[1].squeeze(1).to(args.device)
            cur_token_type_ids = data[2].squeeze(1).to(args.device)
            cur_y = data[3].to(args.device)
            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
            loss = nn.CrossEntropyLoss()(outputs, cur_y.long())
            loss /= args.gradient_accumulation_step
            loss.backward()
            step += 1
            if step % args.gradient_accumulation_step == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            if step % args.report_step == 0:
                accuracy = evaluate(model, dev_loader)
                logger.info('Epoch[{}/{}], loss:{}, accuracy: {}'.format
                            (epoch + 1, args.num_epochs, loss.item(), accuracy))
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('accuracy', accuracy, step)
                model.train()
                if best < accuracy:
                    best = accuracy
                    torch.save(model.state_dict(), join(args.output_path, 'bert.pt'))
                    logger.info('higher_accuracy: {}, step {}, epoch {}, save model\n'.format(best, step, epoch + 1))
    logger.info('dev_best_accuracy: {}'.format(best))


def test(model, args):
    test_data = PrepareDataset(args.test_file, 'valid')
    test_loader = DataLoader(test_data, batch_size=256, shuffle=True)
    model.load_state_dict(torch.load(join(args.output_path, 'bert.pt')))
    accuracy = evaluate(model, test_loader)
    logger.info('test accuracy:{}'.format(accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-max_seq_length', default=256, type=int)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-eval_batch_size', default=512, type=int)
    parser.add_argument('-num_epochs', default=3, type=int)
    parser.add_argument('-learning_rate', default=3e-5, type=float)
    parser.add_argument('-max_grad_norm', default=1.0, type=float)
    parser.add_argument('-warm_up_proportion', default=0.0, type=float)
    parser.add_argument('-gradient_accumulation_step', default=1, type=int)
    parser.add_argument('-bert_path', default='D:/zzj_project/pretrained_model/bert_based_uncased_english', type=str)
    parser.add_argument('-report_step', default=125, type=int)
    parser.add_argument('-output_path', default='output/dataset1', type=str)
    parser.add_argument('-do_train', default=True, type=bool)
    parser.add_argument('-do_test', default=True, type=bool)
    parser.add_argument('-rewrite_cache', default=True, type=bool)
    parser.add_argument('-train_file', default='./pan22/dataset1/train', type=str)
    parser.add_argument('-dev_file', default='pan22/dataset1/validation', type=str)
    parser.add_argument('-test_file', default='pan22/dataset1/validation', type=str)
    args = parser.parse_args()

    # config environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # model initial
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    para_token = ["[PARA_UP]", "[PARA_DOWN]"]
    label_token = ["same", "different"]
    prompt_template = 'They are the [MASK] writing style : {} and {} .'.format(para_token[0], para_token[1])
    special_token_dict = {'additional_special_tokens': para_token}
    tokenizer.add_special_tokens(special_token_dict)
    replace_token_id = tokenizer.convert_tokens_to_ids(label_token)

    # model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels=2)
    # model = Custom_Model(args.bert_path, classification=True)
    model = PromptBERT(args.bert_path)
    # model = torch.nn.DataParallel(model)
    model.to(args.device)
    # save log and tensorboard
    cur_time = time.strftime("%Y%m%d_%H_%M", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)
    writer = SummaryWriter(args.output_path)

    # run
    start_time = time.time()
    if args.do_train:
        train(model, args)
    if args.do_test:
        test(model, args)
    logger.info("run time: {:.4f}".format((time.time() - start_time)/60))
