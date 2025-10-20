import os
import pickle as pkl
import numpy as np
from datetime import datetime
import json
import torch

def _pad_score_to_equal_length(scores, max_docs=20):
    """
    对 tokenized_texts 填充或截断，使其数量达到 max_docs。
    Args:
        tokenized_texts (list of list): 当前样本中的所有文本的索引化序列。
        max_docs (int): 统一的文本序列数量。
    Returns:
        padded_texts (list of list): 填充后的 tokenized_texts。
    """
    max_seq_len = 10
    empty_text = [0] * max_seq_len # 单个空文本的填充值
    # 如果数量不足，填充空文本；如果超出，进行截断
    if len(scores) < max_docs:
        scores.extend([empty_text] * (max_docs - len(scores)))
    else:
        scores = scores[:max_docs]
    return scores

def _pad_texts_to_equal_length1(tokenized_texts, max_docs=20, max_seq_len = 40):
    """
    对 tokenized_texts 填充或截断，使其数量达到 max_docs。
    Args:
        tokenized_texts (list of list): 当前样本中的所有文本的索引化序列。
        max_docs (int): 统一的文本序列数量。
    Returns:
        padded_texts (list of list): 填充后的 tokenized_texts。
    """
    # max_seq_len = 40
    empty_text = [0] * max_seq_len # 单个空文本的填充值
    # 如果数量不足，填充空文本；如果超出，进行截断
    if len(tokenized_texts) < max_docs:
        tokenized_texts.extend([empty_text] * (max_docs - len(tokenized_texts)))
    else:
        tokenized_texts = tokenized_texts[:max_docs]
    return tokenized_texts

def _tokenize_and_pad_text(text, vocab, mode = 'CN'):
    """
    将文本索引化并填充到固定长度。
    """
    max_seq_len = 40
    if mode == 'CN':
        indices = [vocab.get(char, vocab["<unk>"]) for char in text]
    elif mode == "US":
        indices = [vocab.get(char, vocab["<unk>"]) for char in text.split(" ")]
    padded = indices[:max_seq_len] + [vocab["<pad>"]] * max(0, max_seq_len - len(indices))
    return padded

def _pad_texts_to_equal_length(tokenized_texts,vocab, max_docs=20,max_seq_len=40):
    """
    对 tokenized_texts 填充或截断，使其数量达到 max_docs。
    Args:
        tokenized_texts (list of list): 当前样本中的所有文本的索引化序列。
        max_docs (int): 统一的文本序列数量。
    Returns:
        padded_texts (list of list): 填充后的 tokenized_texts。
    """
    # max_seq_len = 40
    empty_text = [vocab["<pad>"]] * max_seq_len # 单个空文本的填充值
    # 如果数量不足，填充空文本；如果超出，进行截断
    if len(tokenized_texts) < max_docs:
        tokenized_texts.extend([empty_text] * (max_docs - len(tokenized_texts)))
    else:
        tokenized_texts = tokenized_texts[:max_docs]
    return tokenized_texts


def _load_stock_data(sequence_dir):
    """
    加载股票序列数据。
    """
    stock_data = {}
    labels = {}
    for sequence_file in os.listdir(sequence_dir):
        path = os.path.join(sequence_dir,sequence_file)
        stock_data[sequence_file.replace(".txt","")] ={}
        labels[sequence_file.replace(".txt","")] ={}
        if not sequence_file.startswith(".ipynb_checkpoints"):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    time = parts[0]
                    features = [float(x) for x in parts[3:6]]
                    stock_data[sequence_file.replace(".txt","")][time] = features
                    features = float(parts[1])
                    labels[sequence_file.replace(".txt","")][time] = features
        # break
    return stock_data, labels

def _load_text_data(directory_path):
    all_data = {}
    for company_dir in os.listdir(directory_path):
        company_path = os.path.join(directory_path, company_dir)
        if os.path.isdir(company_path):
            company_data = {}
            count = 0
            count_num = 0
            for json_file in os.listdir(company_path):
                if json_file.endswith('.json'):
                    file_path = os.path.join(company_path, json_file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if len(data) != 0 and (not (len(data) == 1 and len(data[0]) == 0)):
                                company_data[json_file.replace(".json","")] = data
                                count += len(data[0])
                                count_num += 1
                        except json.JSONDecodeError:
                                print(f"Could not decode JSON from {file_path}")
            all_data[company_dir] = company_data
    return all_data

def _load_text_data2(directory_path):
    all_data = {}
    for company_dir in os.listdir(directory_path):
        company_path = os.path.join(directory_path, company_dir)
        if os.path.isdir(company_path):
            company_data = {}
            count = 0
            count_num = 0
            for json_file in os.listdir(company_path):
                if json_file.endswith('.json'):
                    file_path = os.path.join(company_path, json_file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if len(data) != 0 and (not (len(data) == 1 and len(data["Each_item"]) == 0)):
                                company_data[json_file.replace(".json","")] = data
                                count += len(data["Each_item"])
                                count_num += 1
                        except json.JSONDecodeError:
                                print(f"Could not decode JSON from {file_path}")
            all_data[company_dir] = company_data
    return all_data

def load_cmin_data(data_path, tra_date, val_date, tes_date, seq=5,
                  date_format='%Y-%m-%d',dict = None):
    fnames = [fname for fname in os.listdir(data_path+"/price/preprocessed") if
              os.path.isfile(os.path.join(data_path+"/price/preprocessed", fname))]
    print(len(fnames), ' tickers selected')
    mode = 'CN' if "CN" in data_path else "US"
    print(mode)
    stock_data, labels = _load_stock_data(data_path+"/price/preprocessed")
    dict1 = pkl.load(open("./dict"+"/"+dict, "rb"))
    timeall = set()
    comp_id = {}
    comp1 = 0
    aligned_data_all_train = {}
    aligned_data_all_eval = {}
    aligned_data_all_test = {}
    for comp, times in stock_data.items():
        flag = 0
        if comp in stock_data:
            aligned_data = {}
            for time in times:
                if time in stock_data[comp]:
                    stock_features = stock_data[comp][time]
                    timeall.add(time)
                    date = datetime.strptime(time, '%Y-%m-%d').date()
                    if date >= datetime.strptime(val_date, '%Y-%m-%d').date() and flag == 0:
                        aligned_data_all_train[comp] = aligned_data
                        aligned_data = {}
                        flag = 1
                    if date >= datetime.strptime(tes_date, '%Y-%m-%d').date() and flag == 1:
                        aligned_data_all_eval[comp] = aligned_data
                        aligned_data = {}
                        flag = 2
                    aligned_data[time]={
                            "stock_features": stock_features,
                            "label": labels[comp][time]
                        }
        aligned_data_all_test[comp] = aligned_data
        comp_id[comp] = comp1
        comp1 += 1


    tra_num = 0
    val_num = 0
    tes_num = 0
    fea_dim = 3
    trading_dates = list(timeall)
    trading_dates.sort()
    data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    wd_encodings = np.identity(5, dtype=float)
    dates_index = {}
    dates_dates = {}
    for index, date in enumerate(trading_dates):
        dates_index[date] = index
        dates_dates[index] = date
        data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]
    
    tra_ind = dates_index[tra_date]
    val_ind = dates_index[val_date]
    tes_ind = dates_index[tes_date]

    for comp in aligned_data_all_train:
        count = 0
        for key in aligned_data_all_train[comp]:
            if dates_index[key] <seq:
                continue
            date_ind = dates_index[key]
            map1 = aligned_data_all_train[comp]
            if dates_dates[date_ind - seq] not in map1:
                continue
            tra_num += 1
            count+=1
    for comp in aligned_data_all_eval:
        count = 0
        for key in aligned_data_all_eval[comp]:
            if dates_index[key] <seq:
                continue
            date_ind = dates_index[key]
            map1 = aligned_data_all_train[comp]
            map2 = aligned_data_all_eval[comp]
            if dates_dates[date_ind - seq] not in map1 and dates_dates[date_ind - seq] not in map2:
                continue
            val_num += 1
            count+=1
    for comp in aligned_data_all_test:
        count = 0
        for key in aligned_data_all_test[comp]:
            if dates_index[key] <seq:
                continue
            date_ind = dates_index[key]
            map1 = aligned_data_all_eval[comp]
            map2 = aligned_data_all_test[comp]
            if dates_dates[date_ind - seq] not in map1 and dates_dates[date_ind - seq] not in map2:
                continue
            tes_num += 1
            count+=1
    

    tra_pv = np.zeros([tra_num, seq, fea_dim], dtype=float)
    tra_wd = np.zeros([tra_num, seq, 5], dtype=float)
    tra_gt = np.zeros([tra_num, 1], dtype=float)
    tra_dt = np.zeros([tra_num, 1], dtype=float)
    tra_tn = np.zeros([tra_num, 1], dtype=float)
    tra_txt = np.ones([tra_num, 10, 40],dtype=float) * dict1["<pad>"]
    tra_txt_sc = np.zeros([tra_num,10], dtype=float)
    tra_scores_pad = np.zeros([tra_num,10,10], dtype=float)

    val_pv = np.zeros([val_num, seq, fea_dim], dtype=float)
    val_wd = np.zeros([val_num, seq, 5], dtype=float)
    val_gt = np.zeros([val_num, 1], dtype=float)
    val_dt = np.zeros([val_num, 1], dtype=float)
    val_tn = np.zeros([val_num, 1], dtype=float)
    val_txt = np.ones([val_num, 10, 40], dtype=float) * dict1["<pad>"]
    val_txt_sc = np.zeros([val_num,10], dtype=float)
    val_scores_pad = np.zeros([val_num,10,10], dtype=float)

    tes_pv = np.zeros([tes_num, seq, fea_dim], dtype=float)
    tes_wd = np.zeros([tes_num, seq, 5], dtype=float)
    tes_gt = np.zeros([tes_num, 1], dtype=float)
    tes_dt = np.zeros([tes_num, 1], dtype=float)
    tes_tn = np.zeros([tes_num, 1], dtype=float)
    tes_txt = np.ones([tes_num, 10, 40], dtype=float) * dict1["<pad>"]
    tes_txt_sc = np.zeros([tes_num, 10], dtype=float)
    tes_scores_pad = np.zeros([tes_num,10,10], dtype=float)


    all_text = {}
    all_text_score = {}
    all_pad_score = {}
    text_sc_data_all = _load_text_data(data_path+"/news_score")
    all_text_all = []

    for comp in text_sc_data_all:
        if comp not in all_text:
            all_text[comp_id[comp]] = {}
            all_text_score[comp_id[comp]] = {}
            all_pad_score[comp_id[comp]] = {}
        for date in text_sc_data_all[comp]:
            news_by_date = [b['original_text'] if 'original_text' in b else (b['orginal_text'] if 'orginal_text' in b else "") for b in text_sc_data_all[comp][date][0]]
            Correlation = [[b['scores']['Correlation']] if 'scores' in b and 'Correlation' in b['scores'] else [0.0] for b in text_sc_data_all[comp][date][0]]
            Sentiment = [[b['scores']['Sentiment']] if 'scores' in b and 'Sentiment' in b['scores'] else [0.0] for b in text_sc_data_all[comp][date][0]]
            Importance = [[b['scores']['Importance']] if 'scores' in b and 'Importance' in b['scores'] else [0.0] for b in text_sc_data_all[comp][date][0]]
            Impact = [[b['scores']['Impact']] if 'scores' in b and 'Impact' in b['scores'] else [0.0] for b in text_sc_data_all[comp][date][0]]
            Duration = [[b['scores']['Duration']] if 'scores' in b and 'Duration' in b['scores'] else [0.0] for b in text_sc_data_all[comp][date][0]]
            Entity_Density = [[b['scores']['Entity_Density']] if 'scores' in b and 'Entity_Density' in b['scores'] else [0.0] for b in text_sc_data_all[comp][date][0]]
            Market_Scope = [[b['scores']['Market_Scope']] if 'scores' in b and 'Market_Scope' in b['scores'] else [0.0] for b in text_sc_data_all[comp][date][0]]
            Time_Proximity = [[b['scores']['Time_Proximity']] if 'scores' in b and 'Time_Proximity' in b['scores'] else [0.0] for b in text_sc_data_all[comp][date][0]]
            Headline_Structure = [[b['scores']['Headline_Structure']] if 'scores' in b and 'Headline_Structure' in b['scores'] else [0.0] for b in text_sc_data_all[comp][date][0]]
            Source_Recency = [[b['scores']['Source_Recency']] if 'scores' in b and 'Source_Recency' in b['scores'] else [0.0] for b in text_sc_data_all[comp][date][0]]
            if news_by_date is not None:
                padded_text = [_tokenize_and_pad_text(new,dict1,mode = 'CN' if "CN" in data_path else "US") for new in news_by_date]
                padded_text_pad = _pad_texts_to_equal_length(padded_text,dict1, max_docs = 10,max_seq_len = 40)
            else:
                padded_text = [_tokenize_and_pad_text("None",dict1,mode = 'CN' if "CN" in data_path else "US")]
                padded_text_pad = _pad_texts_to_equal_length([[]], dict1, max_docs = 10,max_seq_len = 40)
        
            scores = [Correlation, Sentiment, Importance, Impact, Duration, Entity_Density, Market_Scope, Time_Proximity, Headline_Structure, Source_Recency]
            scores = [[b[0] for b in sublist] for sublist in scores]
            score_avg = [[sum(b)/len(b) for b in scores]]
            scores = list(map(list, zip(*scores)))
            
            scores_pad = _pad_texts_to_equal_length1(scores,  max_docs = 10,max_seq_len = 10)
            # print(dates_index,date)
            if date in dates_index:
                all_text[comp_id[comp]][dates_index[date]] = np.array(padded_text_pad)
                all_pad_score[comp_id[comp]][dates_index[date]] = np.array(scores_pad)
                all_text_score[comp_id[comp]][dates_index[date]] = np.array(score_avg)

    count = -1
    for comp in aligned_data_all_train:
        for key in aligned_data_all_train[comp]:
            if dates_index[key] < val_ind:
                if dates_index[key] <seq:
                    continue
                date_ind = dates_index[key]
                map1 = aligned_data_all_train[comp]
                if dates_dates[date_ind - seq] not in map1:
                    continue
                count += 1
                tra_pv[count] = np.array([map1[dates_dates[id]]['stock_features'] for id in range(date_ind - seq, date_ind)])
                tra_gt[count] = np.array([1 if (map1[dates_dates[date_ind]]['label'])>0 else 0])
                tra_wd[count] = data_wd[date_ind - seq: date_ind, :]
                tra_dt[count] = dates_index[key]
                tra_tn[count] = comp_id[comp]
                if dates_index[key] in all_text[comp_id[comp]]:
                    tra_txt[count] = np.array(all_text[comp_id[comp]][dates_index[key]],dtype=float)
                    # print(type(all_text_score[comp_id[comp]][dates_index[key]]),type(tra_txt_sc[count]))
                    # print(all_text_score[comp_id[comp]][dates_index[key]].shape,tra_txt_sc[count].shape)
                    tra_txt_sc[count] = all_text_score[comp_id[comp]][dates_index[key]]
                    tra_scores_pad[count] = all_pad_score[comp_id[comp]][dates_index[key]]
                else:
                    tra_txt[count] = np.ones((10,40),dtype=float) * dict1["<pad>"]
                    tra_txt_sc[count] = np.zeros(10,dtype=float)
                    tra_scores_pad[count] = np.ones((10,10),dtype=float) * 0
            else:
                break
    count = -1
    for comp in aligned_data_all_eval:
        for key in aligned_data_all_eval[comp]:
            if val_ind <=dates_index[key] < tes_ind:
                if dates_index[key] <seq:
                    continue
                date_ind = dates_index[key]
                map1 = aligned_data_all_eval[comp]

                if dates_dates[date_ind - seq] not in aligned_data_all_train[comp] and dates_dates[date_ind - seq] not in aligned_data_all_eval[comp]:
                    continue
                count += 1
                val_pv[count] = np.array([map1[dates_dates[id]]['stock_features'] if dates_dates[id] in map1 else aligned_data_all_train[comp][dates_dates[id]]['stock_features'] for id in range(date_ind - seq, date_ind)])
                val_gt[count] = np.array([1 if (map1[dates_dates[date_ind]]['label'])>0 else 0])
                val_wd[count] = data_wd[date_ind - seq: date_ind, :]
                val_dt[count] = dates_index[key]
                val_tn[count] = comp_id[comp]
                if dates_index[key] in all_text[comp_id[comp]]:
                    val_txt[count] = np.array(all_text[comp_id[comp]][dates_index[key]],dtype=float)
                    val_txt_sc[count] = all_text_score[comp_id[comp]][dates_index[key]]
                    val_scores_pad[count] = all_pad_score[comp_id[comp]][dates_index[key]]
                else:
                    val_txt[count] = np.ones((10,40),dtype=float) * dict1["<pad>"]
                    val_txt_sc[count] = np.zeros(10,dtype=float)
                    val_scores_pad[count] = np.ones((10,10),dtype=float) * 0
    count = -1
    for comp in aligned_data_all_test:
        for key in aligned_data_all_test[comp]:
            if tes_ind <= dates_index[key]:
                if dates_index[key] <seq:
                    continue
                date_ind = dates_index[key]
                map1 = aligned_data_all_test[comp]
                if dates_dates[date_ind - seq] not in aligned_data_all_eval[comp] and dates_dates[date_ind - seq] not in aligned_data_all_test[comp]:
                    continue
                count+=1
                tes_pv[count] = np.array([map1[dates_dates[id]]['stock_features'] if dates_dates[id] in map1 else aligned_data_all_eval[comp][dates_dates[id]]['stock_features'] for id in range(date_ind - seq, date_ind)])
                tes_gt[count] = np.array([1 if (map1[dates_dates[date_ind]]['label'])>0 else 0])
                
                tes_wd[count] = data_wd[date_ind - seq: date_ind, :]
                tes_dt[count] = dates_index[key]
                tes_tn[count] = comp_id[comp]
                if dates_index[key] in all_text[comp_id[comp]]:
                    tes_txt[count] = np.array(all_text[comp_id[comp]][dates_index[key]],dtype=float)
                    tes_txt_sc[count] = all_text_score[comp_id[comp]][dates_index[key]]
                    tes_scores_pad[count] = all_pad_score[comp_id[comp]][dates_index[key]]
                else:
                    tes_txt[count] = np.ones((10,40),dtype=float) * dict1["<pad>"]
                    tes_txt_sc[count] = np.zeros(10,dtype=float)
                    tes_scores_pad[count] = np.ones((10,10),dtype=float) * 0


    return tra_pv, tra_gt, tra_wd, tra_dt, tra_tn, tra_txt, tra_txt_sc, tra_scores_pad, val_pv, val_gt, val_wd, val_dt, val_tn, val_txt, val_txt_sc, val_scores_pad, tes_pv, tes_gt, tes_wd, tes_dt, tes_tn, tes_txt, tes_txt_sc, tes_scores_pad


def load_time_data(data_path, tra_date, val_date, tes_date, seq=5,
                  date_format='%Y-%m-%d'):
    fnames = [fname for fname in os.listdir(data_path) if
              os.path.isfile(os.path.join(data_path, fname))]
    
    print(len(fnames), ' tickers selected')

    data_EOD = []
    for index, fname in enumerate(fnames):
        # print(fname)
        single_EOD = np.genfromtxt(
            os.path.join(data_path, fname), dtype=float, delimiter=',',
            skip_header=False
        )
        # print('data shape:', single_EOD.shape)
        data_EOD.append(single_EOD)
    fea_dim = data_EOD[0].shape[1] - 2

    trading_dates = np.genfromtxt(
        os.path.join(data_path, '..', 'trading_dates.csv'), dtype=str,
        delimiter=',', skip_header=False
    )
    print(len(trading_dates), 'trading dates:')

    # transform the trading dates into a dictionary with index, at the same
    # time, transform the indices into a dictionary with weekdays
    dates_index = {}
    # indices_weekday = {}
    data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    wd_encodings = np.identity(5, dtype=float)
    for index, date in enumerate(trading_dates):
        dates_index[date] = index
        # indices_weekday[index] = datetime.strptime(date, date_format).weekday()
        data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]

    tra_ind = dates_index[tra_date]
    val_ind = dates_index[val_date]
    tes_ind = dates_index[tes_date]
    print(tra_ind, val_ind, tes_ind)

    # count training, validation, and testing instances
    tra_num = 0
    val_num = 0
    tes_num = 0
    # training
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    tra_num += 1
    print(tra_num, ' training instances')

    # validation
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    val_num += 1
    print(val_num, ' validation instances')

    # testing
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    tes_num += 1
    print(tes_num, ' testing instances')

    # generate training, validation, and testing instances
    # training
    tra_pv = np.zeros([tra_num, seq, fea_dim], dtype=float)
    tra_wd = np.zeros([tra_num, seq, 5], dtype=float)
    tra_gt = np.zeros([tra_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                    data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tra_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, : -2]
                tra_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tra_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1

    # validation
    val_pv = np.zeros([val_num, seq, fea_dim], dtype=float)
    val_wd = np.zeros([val_num, seq, 5], dtype=float)
    val_gt = np.zeros([val_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                            data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                val_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                val_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                val_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1

    # testing
    tes_pv = np.zeros([tes_num, seq, fea_dim], dtype=float)
    tes_wd = np.zeros([tes_num, seq, 5], dtype=float)
    tes_gt = np.zeros([tes_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                            data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tes_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                # # for the momentum indicator
                # tes_pv[ins_ind, -1, -1] = data_EOD[tic_ind][date_ind - 1, -1] - data_EOD[tic_ind][date_ind - 11, -1]
                tes_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tes_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1
    return tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt
def load_stocknet_data(data_path, tra_date, val_date, tes_date, seq=5,
                  date_format='%Y-%m-%d',dict = None):
    fnames = [fname for fname in os.listdir(data_path+"/price/preprocessed") if
              os.path.isfile(os.path.join(data_path+"/price/preprocessed", fname))]
    print(len(fnames), ' tickers selected')
    mode = 'CN' if "CN" in data_path else "US"
    print(mode)
    stock_data, labels = _load_stock_data(data_path+"/price/preprocessed")
    dict1 = pkl.load(open("./dict"+"/"+dict, "rb"))
    timeall = set()
    comp_id = {}
    comp1 = 0
    aligned_data_all_train = {}
    aligned_data_all_eval = {}
    aligned_data_all_test = {}
    for comp, times in stock_data.items():
        flag = 0
        if comp in stock_data:
            aligned_data = {}
            for time in times:
                if time in stock_data[comp]:
                    stock_features = stock_data[comp][time]
                    timeall.add(time)
                    date = datetime.strptime(time, '%Y-%m-%d').date()
                    if date >= datetime.strptime(val_date, '%Y-%m-%d').date() and flag == 0:
                        aligned_data_all_train[comp] = aligned_data
                        aligned_data = {}
                        flag = 1
                    if date >= datetime.strptime(tes_date, '%Y-%m-%d').date() and flag == 1:
                        aligned_data_all_eval[comp] = aligned_data
                        aligned_data = {}
                        flag = 2
                    aligned_data[time]={
                            "stock_features": stock_features,
                            "label": labels[comp][time]
                        }
        aligned_data_all_test[comp] = aligned_data
        comp_id[comp] = comp1
        comp1 += 1


    tra_num = 0
    val_num = 0
    tes_num = 0
    fea_dim = 3
    trading_dates = list(timeall)
    trading_dates.sort()
    data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    wd_encodings = np.identity(5, dtype=float)
    dates_index = {}
    dates_dates = {}
    for index, date in enumerate(trading_dates):
        dates_index[date] = index
        dates_dates[index] = date
        data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]
    
    tra_ind = dates_index[tra_date]
    val_ind = dates_index[val_date]
    tes_ind = dates_index[tes_date]

    for comp in aligned_data_all_train:
        count = 0
        for key in aligned_data_all_train[comp]:
            if dates_index[key] <seq:
                continue
            date_ind = dates_index[key]
            map1 = aligned_data_all_train[comp]
            if dates_dates[date_ind - seq] not in map1:
                continue
            tra_num += 1
            count+=1
    for comp in aligned_data_all_eval:
        count = 0
        for key in aligned_data_all_eval[comp]:
            if dates_index[key] <seq:
                continue
            date_ind = dates_index[key]
            map1 = aligned_data_all_train[comp]
            map2 = aligned_data_all_eval[comp]
            if dates_dates[date_ind - seq] not in map1 and dates_dates[date_ind - seq] not in map2:
                continue
            val_num += 1
            count+=1
    for comp in aligned_data_all_test:
        count = 0
        for key in aligned_data_all_test[comp]:
            if dates_index[key] <seq:
                continue
            date_ind = dates_index[key]
            map1 = aligned_data_all_eval[comp]
            map2 = aligned_data_all_test[comp]
            if dates_dates[date_ind - seq] not in map1 and dates_dates[date_ind - seq] not in map2:
                continue
            tes_num += 1
            count+=1
    

    tra_pv = np.zeros([tra_num, seq, fea_dim], dtype=float)
    tra_wd = np.zeros([tra_num, seq, 5], dtype=float)
    tra_gt = np.zeros([tra_num, 1], dtype=float)
    tra_dt = np.zeros([tra_num, 1], dtype=float)
    tra_tn = np.zeros([tra_num, 1], dtype=float)
    tra_txt = np.ones([tra_num, 10, 40],dtype=float) * dict1["<pad>"]
    tra_txt_sc = np.zeros([tra_num,10], dtype=float)
    tra_scores_pad = np.zeros([tra_num,10,10], dtype=float)

    val_pv = np.zeros([val_num, seq, fea_dim], dtype=float)
    val_wd = np.zeros([val_num, seq, 5], dtype=float)
    val_gt = np.zeros([val_num, 1], dtype=float)
    val_dt = np.zeros([val_num, 1], dtype=float)
    val_tn = np.zeros([val_num, 1], dtype=float)
    val_txt = np.ones([val_num, 10, 40], dtype=float) * dict1["<pad>"]
    val_txt_sc = np.zeros([val_num,10], dtype=float)
    val_scores_pad = np.zeros([val_num,10,10], dtype=float)

    tes_pv = np.zeros([tes_num, seq, fea_dim], dtype=float)
    tes_wd = np.zeros([tes_num, seq, 5], dtype=float)
    tes_gt = np.zeros([tes_num, 1], dtype=float)
    tes_dt = np.zeros([tes_num, 1], dtype=float)
    tes_tn = np.zeros([tes_num, 1], dtype=float)
    tes_txt = np.ones([tes_num, 10, 40], dtype=float) * dict1["<pad>"]
    tes_txt_sc = np.zeros([tes_num, 10], dtype=float)
    tes_scores_pad = np.zeros([tes_num,10,10], dtype=float)


    all_text = {}
    all_text_score = {}
    all_pad_score = {}
    text_sc_data_all = _load_text_data2(data_path+"/score")
    all_text_all = []

    for comp in text_sc_data_all:
        if comp not in all_text:
            all_text[comp_id[comp]] = {}
            all_text_score[comp_id[comp]] = {}
            all_pad_score[comp_id[comp]] = {}
        
        for date in text_sc_data_all[comp]:
            news_by_date = [b['News_content'] if 'News_content' in b else (b['orginal_text'] if 'orginal_text' in b else "") for b in text_sc_data_all[comp][date]["Each_item"]]
            Correlation = [[b['Correlation']] if  'Correlation' in b else [0.0] for b in text_sc_data_all[comp][date]["Each_item"]]
            Sentiment = [[b['Sentiment']] if 'Sentiment' in b else [0.0] for b in text_sc_data_all[comp][date]["Each_item"]]
            Importance = [[b['Importance']] if 'Importance' in b else [0.0] for b in text_sc_data_all[comp][date]["Each_item"]]
            Impact = [[b['Impact']] if 'Impact' in b else [0.0] for b in text_sc_data_all[comp][date]["Each_item"]]
            Duration = [[b['Duration']] if 'Duration' in b else [0.0] for b in text_sc_data_all[comp][date]["Each_item"]]
            Entity_Density = [[b['Virality']] if 'Virality' in b else [0.0] for b in text_sc_data_all[comp][date]["Each_item"]]
            Market_Scope = [[b['Source_Score']] if 'Source_Score' in b else [0.0] for b in text_sc_data_all[comp][date]["Each_item"]]
            Time_Proximity = [[b['Specificity']] if 'Specificity' in b else [0.0] for b in text_sc_data_all[comp][date]["Each_item"]]
            Headline_Structure = [[b['Sector_Spread']] if 'Sector_Spread' in b else [0.0] for b in text_sc_data_all[comp][date]["Each_item"]]
            Source_Recency = [[b['Tech_mentions']] if 'Tech_mentions' in b else [0.0] for b in text_sc_data_all[comp][date]["Each_item"]]
            if len(news_by_date) == 0:
                continue
            if news_by_date is not None:
                padded_text = [_tokenize_and_pad_text(new,dict1,mode = 'CN' if "CN" in data_path else "US") for new in news_by_date]
                padded_text_pad = _pad_texts_to_equal_length(padded_text,dict1, max_docs = 10,max_seq_len = 40)
            else:
                padded_text = [_tokenize_and_pad_text("None",dict1,mode = 'CN' if "CN" in data_path else "US")]
                padded_text_pad = _pad_texts_to_equal_length([[]], dict1, max_docs = 10,max_seq_len = 40)
            
            scores = [Correlation, Sentiment, Importance, Impact, Duration, Entity_Density, Market_Scope, Time_Proximity, Headline_Structure, Source_Recency]
            scores = [[b[0] for b in sublist] for sublist in scores]
            
            score_avg = [[sum(b)/len(b) for b in scores]]
            scores = list(map(list, zip(*scores)))
            
            scores_pad = _pad_texts_to_equal_length1(scores,  max_docs = 10,max_seq_len = 10)
            # print(dates_index,date)
            if date in dates_index:
                all_text[comp_id[comp]][dates_index[date]] = np.array(padded_text_pad)
                all_pad_score[comp_id[comp]][dates_index[date]] = np.array(scores_pad)
                all_text_score[comp_id[comp]][dates_index[date]] = np.array(score_avg)

    count = -1
    for comp in aligned_data_all_train:
        for key in aligned_data_all_train[comp]:
            if dates_index[key] < val_ind:
                if dates_index[key] <seq:
                    continue
                date_ind = dates_index[key]
                map1 = aligned_data_all_train[comp]
                if dates_dates[date_ind - seq] not in map1:
                    continue
                count += 1
                tra_pv[count] = np.array([map1[dates_dates[id]]['stock_features'] for id in range(date_ind - seq, date_ind)])
                tra_gt[count] = np.array([1 if (map1[dates_dates[date_ind]]['label'])>0 else 0])
                tra_wd[count] = data_wd[date_ind - seq: date_ind, :]
                tra_dt[count] = dates_index[key]
                tra_tn[count] = comp_id[comp]
                if dates_index[key] in all_text[comp_id[comp]]:
                    tra_txt[count] = np.array(all_text[comp_id[comp]][dates_index[key]],dtype=float)
                    tra_txt_sc[count] = all_text_score[comp_id[comp]][dates_index[key]]
                    tra_scores_pad[count] = all_pad_score[comp_id[comp]][dates_index[key]]
                else:
                    tra_txt[count] = np.ones((10,40),dtype=float) * dict1["<pad>"]
                    tra_txt_sc[count] = np.zeros(10,dtype=float)
                    tra_scores_pad[count] = np.ones((10,10),dtype=float) * 0
            else:
                break
    count = -1
    for comp in aligned_data_all_eval:
        for key in aligned_data_all_eval[comp]:
            if val_ind <=dates_index[key] < tes_ind:
                if dates_index[key] <seq:
                    continue
                date_ind = dates_index[key]
                map1 = aligned_data_all_eval[comp]

                if dates_dates[date_ind - seq] not in aligned_data_all_train[comp] and dates_dates[date_ind - seq] not in aligned_data_all_eval[comp]:
                    continue
                count += 1
                val_pv[count] = np.array([map1[dates_dates[id]]['stock_features'] if dates_dates[id] in map1 else aligned_data_all_train[comp][dates_dates[id]]['stock_features'] for id in range(date_ind - seq, date_ind)])
                val_gt[count] = np.array([1 if (map1[dates_dates[date_ind]]['label'])>0 else 0])
                val_wd[count] = data_wd[date_ind - seq: date_ind, :]
                val_dt[count] = dates_index[key]
                val_tn[count] = comp_id[comp]
                if dates_index[key] in all_text[comp_id[comp]]:
                    val_txt[count] = np.array(all_text[comp_id[comp]][dates_index[key]],dtype=float)
                    val_txt_sc[count] = all_text_score[comp_id[comp]][dates_index[key]]
                    val_scores_pad[count] = all_pad_score[comp_id[comp]][dates_index[key]]
                else:
                    val_txt[count] = np.ones((10,40),dtype=float) * dict1["<pad>"]
                    val_txt_sc[count] = np.zeros(10,dtype=float)
                    val_scores_pad[count] = np.ones((10,10),dtype=float) * 0
    count = -1
    for comp in aligned_data_all_test:
        for key in aligned_data_all_test[comp]:
            if tes_ind <= dates_index[key]:
                if dates_index[key] <seq:
                    continue
                date_ind = dates_index[key]
                map1 = aligned_data_all_test[comp]
                if dates_dates[date_ind - seq] not in aligned_data_all_eval[comp] and dates_dates[date_ind - seq] not in aligned_data_all_test[comp]:
                    continue
                count+=1
                # if dates_dates[id] in map1:
                # print(map1, comp,aligned_data_all_eval[comp])
                tes_pv[count] = np.array([map1[dates_dates[id]]['stock_features'] if dates_dates[id] in map1 else (aligned_data_all_eval[comp][dates_dates[id]]['stock_features'] if dates_dates[id] in aligned_data_all_eval[comp] else map1[dates_dates[id+1]]['stock_features']) for id in range(date_ind - seq, date_ind)])
                tes_gt[count] = np.array([1 if (map1[dates_dates[date_ind]]['label'])>0 else 0])
                
                tes_wd[count] = data_wd[date_ind - seq: date_ind, :]
                tes_dt[count] = dates_index[key]
                tes_tn[count] = comp_id[comp]
                if comp_id[comp] in all_text:
                    if dates_index[key] in all_text[comp_id[comp]]:
                        tes_txt[count] = np.array(all_text[comp_id[comp]][dates_index[key]],dtype=float)
                        tes_txt_sc[count] = all_text_score[comp_id[comp]][dates_index[key]]
                        tes_scores_pad[count] = all_pad_score[comp_id[comp]][dates_index[key]]
                    else:
                        tes_txt[count] = np.ones((10,40),dtype=float) * dict1["<pad>"]
                        tes_txt_sc[count] = np.zeros(10,dtype=float)
                        tes_scores_pad[count] = np.ones((10,10),dtype=float) * 0


    return tra_pv, tra_gt, tra_wd, tra_dt, tra_tn, tra_txt, tra_txt_sc, tra_scores_pad, val_pv, val_gt, val_wd, val_dt, val_tn, val_txt, val_txt_sc, val_scores_pad, tes_pv, tes_gt, tes_wd, tes_dt, tes_tn, tes_txt, tes_txt_sc, tes_scores_pad
    


data_period = {
    "kdd17": {
        "train_start": "2007-01-03", "train_end": "2015-01-01",  
        "val_start": "2015-01-02", "val_end": "2016-01-03",  
        "test_start": "2016-01-04", "test_end": "2017-01-01"
    },
    "ni225": {
        "train_start": "2016-07-01", "train_end": "2018-03-01",  
        "val_start": "2018-03-02", "val_end": "2019-01-06",  
        "test_start": "2019-01-07", "test_end": "2019-12-31"
    },
    "ftse100": {
        "train_start": "2014-01-06", "train_end": "2017-01-03",  
        "val_start": "2017-01-04", "val_end": "2017-07-03",  
        "test_start": "2017-07-04", "test_end": "2018-06-30"
    },
    "csi300": {
        "train_start": "2015-06-02", "train_end": "2017-02-28",  
        "val_start": "2017-03-01", "val_end": "2018-03-01",  
        "test_start": "2018-03-02", "test_end": "2019-12-31"
    },
    "stocknet": {
        "train_start": "2014-01-02", "train_end": "2015-08-02",  
        "val_start": "2015-08-03", "val_end": "2015-09-30",  
        "test_start": "2015-10-01", "test_end": "2016-01-01",
        "dict": "dict_tweet.pkl"
    },
    "CMIN-CN": {
        "train_start": "2018-01-03",
        "train_end": "2021-04-30",
        "val_start": "2021-05-06",
        "val_end": "2021-08-31",
        "test_start": "2021-09-01",
        "test_end": "2021-12-31",
        "dict": "dict_cn.pkl"
    },
    "CMIN-US": {
        "train_start": "2018-01-03",
        "train_end": "2021-04-30",
        "val_start": "2021-05-06",
        "val_end": "2021-08-31",
        "test_start": "2021-09-01",
        "test_end": "2021-12-31",
        "dict": "dict_us.pkl"
    }
}
window = 5
path = "/home/zhaokx/formal_from_autodl/finance_stock_datasets"
datasets = ["kdd17","ni225","ftse100","csi300"]
datasets = ["CMIN-US","CMIN-CN"]
datasets = ["stocknet"] 
for dir1 in datasets:
    if dir1.startswith("CMIN"):
        path1 = os.path.join(path, dir1)
        tra_pv, tra_gt, tra_wd, tra_dt, tra_tn, tra_txt, tra_txt_sc, tra_scores_pad, val_pv, val_gt, val_wd, val_dt, val_tn, val_txt, val_txt_sc, val_scores_pad, tes_pv, tes_gt, tes_wd, tes_dt, tes_tn, tes_txt, tes_txt_sc, tes_scores_pad = load_cmin_data(
            path1,
            data_period[dir1]["train_start"], data_period[dir1]["val_start"], data_period[dir1]["test_start"],seq=window, dict = data_period[dir1]["dict"]
        )
        np.savez_compressed(dir1+'_data.npz',  tra_pv=tra_pv, tra_gt=tra_gt, tra_wd=tra_wd, tra_dt=tra_dt, tra_co=tra_tn, tra_txt=tra_txt, tra_txt_sc=tra_txt_sc, tra_scores_pad=tra_scores_pad, val_pv=val_pv, val_gt=val_gt, val_wd=val_wd, val_dt=val_dt, val_co=val_tn, val_txt=val_txt, val_txt_sc=val_txt_sc, val_scores_pad=val_scores_pad, tes_pv=tes_pv, tes_gt=tes_gt, tes_wd=tes_wd, tes_dt=tes_dt, tes_co=tes_tn, tes_txt=tes_txt, tes_txt_sc=tes_txt_sc, tes_scores_pad=tes_scores_pad
        )
    elif dir1 != "stocknet": 
        path1 = os.path.join(path,dir1,"ourpped")
        tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt = load_time_data(
            path1,
            data_period[dir1]["train_start"], data_period[dir1]["val_start"], data_period[dir1]["test_start"],seq=window
        )
        np.savez_compressed(dir1+'_data.npz', tes_pv=tes_pv,tes_wd=tes_wd,tes_gt=tes_gt, tra_pv=tra_pv, tra_wd=tra_wd, tra_gt=tra_gt,val_pv=val_pv, val_gt=val_gt, val_wd = val_wd
        )
    else:
        path1 = os.path.join(path,dir1)
        tra_pv, tra_gt, tra_wd, tra_dt, tra_tn, tra_txt, tra_txt_sc, tra_scores_pad, val_pv, val_gt, val_wd, val_dt, val_tn, val_txt, val_txt_sc, val_scores_pad, tes_pv, tes_gt, tes_wd, tes_dt, tes_tn, tes_txt, tes_txt_sc, tes_scores_pad = load_stocknet_data(
            path1,
            data_period[dir1]["train_start"], data_period[dir1]["val_start"], data_period[dir1]["test_start"],seq=window,  dict = data_period[dir1]["dict"]
        )
        np.savez_compressed(dir1+'_data.npz', tra_pv=tra_pv, tra_gt=tra_gt, tra_wd=tra_wd, tra_dt=tra_dt, tra_co=tra_tn, tra_txt=tra_txt, tra_txt_sc=tra_txt_sc, tra_scores_pad=tra_scores_pad, val_pv=val_pv, val_gt=val_gt, val_wd=val_wd, val_dt=val_dt, val_co=val_tn, val_txt=val_txt, val_txt_sc=val_txt_sc, val_scores_pad=val_scores_pad, tes_pv=tes_pv, tes_gt=tes_gt, tes_wd=tes_wd, tes_dt=tes_dt, tes_co=tes_tn, tes_txt=tes_txt, tes_txt_sc=tes_txt_sc, tes_scores_pad=tes_scores_pad
        )
       