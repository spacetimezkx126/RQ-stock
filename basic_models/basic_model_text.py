import os
import torch
import optuna
import math
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from dtml_trans import DTML_trans
from mefi import MambaTimeSeriesModel
import random
from collections import defaultdict
def calculate_mcc(y_true, y_pred):
    """
    计算 MCC (Matthew's Correlation Coefficient) 指标。
    
    Args:
        y_true (list or np.ndarray): 真实标签。
        y_pred (list or np.ndarray): 预测标签。
    
    Returns:
        float: MCC 值。
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator != 0 else 0.0

def set_seed(seed=42):
    torch.manual_seed(seed)  # 为 CPU 设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU，设置所有 GPU 的随机种子
    np.random.seed(seed)  # 设置 numpy 的随机种子
    random.seed(seed)  # 设置 python 的随机种子
    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法一致
    torch.backends.cudnn.benchmark = False  # 如果输入数据的大小不变，设置为 False 可能会提升性能

class Basic_Model_Fused(nn.Module):
    def __init__(self, dims):
        super(Basic_Model_Fused, self).__init__()
        embed = 200
        self.embedding = nn.Embedding(186100, embed)
        
        self.mlp = nn.Sequential(nn.Linear(296,embed),nn.LeakyReLU(0.03),nn.Linear(embed,150),nn.LeakyReLU(0.03),nn.Linear(150,50),nn.LeakyReLU(0.03),nn.Linear(50,1))
        self.score_emb = nn.Linear(10,50)
        self.window_size = 5
        self.stock_lstm = nn.LSTM(146, embed, batch_first=True, bidirectional=False)
        self.text_fc = nn.Linear(144, hidden_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 48, (k, embed)) for k in (2, 3, 4)]
        )
        self.mlp = nn.Sequential(nn.Linear(128,32),nn.ReLU(),nn.Linear(32,1))
        self.conv = nn.Conv1d(
            in_channels=20 * hidden_dim,  # 合并 num_docs 和 hidden_dim
            out_channels=hidden_dim,  # 输出通道
            kernel_size=3,
            padding=3 // 2  # 保持长度不变
        )
        self.pool = nn.MaxPool1d(2)
    def conv_and_pool(self, x, conv):
        """卷积 + 最大池化"""
        x = F.relu(conv(x)).squeeze(3)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    def forward(self, x_emb, texts, score):
        texts = texts.squeeze(0)
        batch_size, num_docs, max_words = texts.size()
        texts = texts.reshape(-1, max_words)
        xtext = self.embedding(texts.to(device)).unsqueeze(1)
        text_repr = torch.cat([self.conv_and_pool(xtext, conv) for conv in self.convs], 1)
        text_repr2 = self.text_fc(text_repr)
        text_repr = text_repr2.view( 1, batch_size, num_docs, -1)
        
        text_repr = text_repr.view( 1,  batch_size, num_docs * text_repr2.shape[-1]).permute(0,2,1) 
        output = self.conv(text_repr).permute(0,2,1)
        output = self.pool(output)
        final_predictions = self.mlp(torch.cat([x_emb,output],dim=-1))
        return final_predictions

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.0):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.mefi = MambaTimeSeriesModel(16,64)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, x = self.mefi(x)
        out, _ = self.lstm(x)             # out: (batch, seq_len, hidden_size)
        last = out[:, -1, :]              # 取最后一个时间步
        logits = self.fc(last).squeeze(-1)# (batch,)
        return logits

def test(expanded_label,expanded_label1,expanded_label2,expanded_txt1,expanded_txt2,expanded_tensor1,expanded_tensor2, expanded_txt_sc, expanded_txt1_sc, expanded_txt2_sc):
    
    layers = 2
    heads = 4
    dims = 32
    pos_weis = 1
    learning_rate = 0.0005
    mamba_ly = 1
    # 2 3 32 1.0 0.0005 0.1
    gl_wei = 0.3
    dtml = torch.load("/home/zhaokx/stock_text_pred/RQ-stock/checkpoints/cmin_cn/2_3_32_0.7_1e-05_2_0.3_model_time_b.pth",weights_only=False,map_location=device).to(device)
    # dtml = torch.load("/home/zhaokx/stock_text_pred/RQ-stock/checkpoints/cmin_cn/3_4_32_1.0_5e-05_2_0.3_model_time_b.pth",weights_only=False,map_location=device).to(device)
    context_model = FuseContext(dims).to(device)
    # context_model = torch.load("context_model.pth", weights_only=False, map_location=device)
    optimizer = optim.Adam(dtml.parameters(), lr=learning_rate)
    optimizer1 = optim.RMSprop(context_model.parameters(), lr=0.00001)
    # ftse 0.1, 16, 32, max1+1, 2, 2 0.8
    # ni225 0.1, 16, 32, max1+1, 2, 1 0.85
    # kdd17 0.1, 16, 32, max1+1, 2, 2 0.8
    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([pos_weis]).to(device))
    idx = 10000
    best_eval = 0
    best = 0
    pad_vector = torch.full((40,), 59041).to(device) 
    for epoch in range(num_epochs):
        print(epoch)
        correct_test = 0
        dtml.eval()
        context_model.train()
        correct_eval = 0
        count = 0
        emb_out = None
        temp = -1
        torch.cuda.empty_cache()
        last = 0
        for i, (date, samples) in enumerate(grouped_data.items()):
            if i % len(expanded_tensor) == 0 and i!=0:
                expanded_tensor_b = expanded_tensor.permute(0,2,1,3)
                expanded_txt_b = expanded_txt.permute(1,0,2,3)
                expanded_txt_sc_b = expanded_txt_sc.permute(1,0,2)
                for j in range(0,len(expanded_tensor_b),idx):
                    data_train = []
                    label_train = []
                    dt_train = expanded_tensor_b[j:j+idx,:,:].to(device)
                    dt_text = expanded_txt[j:j+idx,:,:,:].to(device)
                    pred, train_emb = dtml(dt_train)
                    dt_label = expanded_label[j:j+idx,:].to(device)
                    lb_mask = []
                    for k in range(len(dt_label)):
                        data_train.append(pred[k][(dt_label[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                        label_train.append(dt_label[k][(dt_label[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                    data_train1 = torch.cat(data_train, dim=0)
                    label_train1 = torch.cat(label_train, dim=0).squeeze(-1)
                    predicted1 = (torch.sigmoid(data_train1) > 0.5).int()
                    dt_label = dt_label.permute(1,0,2)
                    train_emb = train_emb.permute(1,0,2)
                    dt_text = dt_text.permute(1,0,2,3)
                    for o in range(len(dt_label)):
                        optimizer1.zero_grad()
                        out = context_model(train_emb[o].detach().unsqueeze(0), dt_text[o].unsqueeze(0), expanded_txt_sc_b[o].unsqueeze(0).to(device))
                        label = dt_label[o]
                        loss = criterion(out[0][(dt_label[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)], label[(dt_label[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                        predicted = (torch.sigmoid(out[0][(dt_label[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)]) > 0.5).int()
                        loss.backward()
                        optimizer1.step()
            i1 = i % len(expanded_tensor)
            expanded_tensor[i1, grouped_sid[date]] = samples.float()
            expanded_label[i1,grouped_sid[date]] = grouped_label[date].float()
            expanded_txt[i1, grouped_sid[date]] = grouped_txt[date]
            expanded_txt_sc[i1, grouped_sid[date]] = grouped_txt_sc[date].float()
            last = i1+1
        dtml.eval()
        context_model.eval()

        with torch.no_grad():
            temp = -1
            correct3 = 0
            predicted1 = 0
            countall1 = 0
            countall2 = 0
            correct4 = 0
            label1 = 0
            label11 = 0
            test_num_count = 0
            last = 0
            predict_all_eval = []
            label_all_eval = []
            for j, (date, samples) in enumerate(grouped_data1.items()):
                if j % len(expanded_tensor1) == 0 and j!=0:
                    expanded_tensor1t = expanded_tensor1.permute(0,2,1,3)
                    data_eval = []
                    label_eval = []
                    pred, eval_emb = dtml(expanded_tensor1t.to(device))
                    for k in range(len(expanded_label1)):
                        data_eval.append(pred[k][(expanded_label1[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                        label_eval.append(expanded_label1[k].to(device)[(expanded_label1[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                    data_eval1 = torch.cat(data_eval, dim=0).to(device)
                    label_eval1 = torch.cat(label_eval, dim=0).squeeze(-1).to(device)
                    loss1 = criterion(data_eval1.squeeze(), label_eval1.float())
                    predicted1 = (torch.sigmoid(data_eval1) > 0.5).int()
                    correct3 += (label_eval1.to(device) == predicted1.to(device).squeeze(1)).sum().item()
                    countall1 += len(label_eval1)
                    label1 += sum(label_eval1)
                    eval_emb = eval_emb.permute(1,0,2)
                    expanded_label11 = expanded_label1.permute(1,0,2)
                    expanded_txt_b = expanded_txt1.permute(1,0,2,3)
                    expanded_txt_sc_b = expanded_txt1_sc.permute(1,0,2)
                    pred = []
                    for o in range(len(expanded_label11)):
                        out = context_model(eval_emb[o].detach().unsqueeze(0), expanded_txt_b[o].unsqueeze(0), expanded_txt_sc_b[o].unsqueeze(0).to(device))
                        label = expanded_label11[o]
                        loss = criterion(out[0][(expanded_label11[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].to(device), label.to(device)[(expanded_label11[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                        predicted = (torch.sigmoid(out[0][(expanded_label11[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].to(device)) > 0.5).int()
                        correct_eval += (label.to(device)[(expanded_label11[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].squeeze(1).to(device) == predicted.squeeze(1)).sum().item()
                        pred.append(out[0][(expanded_label11[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].to(device))
                

                i1 = j % len(expanded_tensor1)
                expanded_tensor1[i1, grouped_sid1[date]] = samples.float()
                expanded_label1[i1, grouped_sid1[date]] = grouped_label1[date].float()
                expanded_txt1[i1, grouped_sid1[date]] = grouped_txt1[date]
                expanded_ratio1[i1, grouped_sid1[date]] = grouped_ratio1[date].float()
                expanded_sum1[i1,grouped_sid1[date]] = grouped_sum1[date].float()
                expanded_txt1_sc[i1, grouped_sid1[date]] = grouped_txt1_sc[date].float()
                last = i1+1
            expanded_tensor1 = expanded_tensor1[:last, :,:,:]
            expanded_label1 = expanded_label1[:last,:,:]
            expanded_txt1 = expanded_txt1[:last,:,:,:]
            expanded_txt1_sc = expanded_txt1_sc[:last,:,:]


            expanded_tensor1t = expanded_tensor1.permute(0,2,1,3)
            data_eval = []
            label_eval = []
            pred, eval_emb = dtml(expanded_tensor1t.to(device))
            for k in range(len(expanded_label1)):
                data_eval.append(pred[k][(expanded_label1[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                label_eval.append(expanded_label1[k].to(device)[(expanded_label1[k].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
            data_eval1 = torch.cat(data_eval, dim=0).to(device)
            label_eval1 = torch.cat(label_eval, dim=0).squeeze(-1).to(device)
            loss1 = criterion(data_eval1.squeeze(), label_eval1.float())
            predicted1 = (torch.sigmoid(data_eval1) > 0.5).int()
            correct3 += (label_eval1.to(device) == predicted1.to(device).squeeze(1)).sum().item()
            countall1 += len(label_eval1)
            label1 += sum(label_eval1)
            eval_emb = eval_emb.permute(1,0,2)
            expanded_label11 = expanded_label1.permute(1,0,2)
            expanded_txt_b = expanded_txt1.permute(1,0,2,3)
            expanded_txt_sc_b = expanded_txt1_sc.permute(1,0,2)
            pred = []
            for o in range(len(expanded_label11)):
                out = context_model(eval_emb[o].detach().unsqueeze(0), expanded_txt_b[o].unsqueeze(0), expanded_txt_sc_b[o].unsqueeze(0).to(device))
                label = expanded_label11[o]
                loss = criterion(out[0][(expanded_label11[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].to(device), label.to(device)[(expanded_label11[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                predicted = (torch.sigmoid(out[0][(expanded_label11[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].to(device)) > 0.5).int()
                correct_eval += (label.to(device)[(expanded_label11[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].squeeze(1).to(device) == predicted.squeeze(1)).sum().item()
                pred.append(out[0][(expanded_label11[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].to(device))
                predict_all_eval.append(predicted.squeeze(1))
                label_all_eval.append(label.to(device)[(expanded_label11[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].squeeze(1))
            temp = -1
            correct2 = 0
            predict_all = []
            label_all = []
            
            label_temp = None
            last = 0
            for j, (date, samples) in enumerate(grouped_data2.items()):
                if j % len(expanded_tensor2) == 0 and j!=0:
                    expanded_tensor2t = expanded_tensor2.permute(0,2,1,3)
                    expanded_txt_b = expanded_txt2.permute(1,0,2,3)
                    expanded_txt_sc_b = expanded_txt2_sc.permute(1,0,2)
                    pred, test_emb = dtml(expanded_tensor2t.to(device))
                    data_test = []
                    label_test = []
                    test_num = []
                    for k in range(len(expanded_label2)):
                        data_test.append(pred[k][(expanded_label2[k]!=torch.tensor(2)).squeeze(1).to(device)])
                        label_test.append(expanded_label2[k].to(device)[(expanded_label2[k]!=torch.tensor(2)).squeeze(1).to(device)])
                        test_num.append(pred[k])
                    test_num = torch.cat(test_num, dim=0)
                    test_num_count += len(test_num)
                    data_test2 = torch.cat(data_test, dim=0)
                    label_test2 = torch.cat(label_test, dim=0).squeeze(-1)
                    countall2 += len(label_test2)
                    predicted1 = (torch.sigmoid(data_test2) > 0.5).int()
                    correct4 += (label_test2.to(device) == predicted1.to(device).squeeze(1)).sum().item()
                    test_emb = test_emb.permute(1,0,2)
                    expanded_label21 = expanded_label2.permute(1,0,2)
                    pred = []
                    for o in range(len(expanded_label21)):
                        out = context_model(test_emb[o].detach().unsqueeze(0), expanded_txt_b[o].unsqueeze(0), expanded_txt_sc_b[o].unsqueeze(0).to(device))
                        label = expanded_label21[o].clone()
                        loss = criterion(out[0][(expanded_label21[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].to(device), label.to(device)[(expanded_label21[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                        predicted = (torch.sigmoid(out[0][(expanded_label21[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].to(device)) > 0.5).int()
                        correct_test += (label.to(device)[(expanded_label21[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].squeeze(1).to(device) == predicted.squeeze(1)).sum().item()
                        predict_all.append(predicted.squeeze(1))
                        label_all.append(label.to(device)[(expanded_label21[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].squeeze(1))
                        label_temp = label_all
                i1 = j % len(expanded_tensor2)
                expanded_tensor2[i1, grouped_sid2[date]] = samples.float()
                expanded_label2[i1, grouped_sid2[date]] = grouped_label2[date].float()
                expanded_txt2[i1, grouped_sid2[date]] = grouped_txt2[date]
                expanded_ratio2[i1, grouped_sid2[date]] = grouped_ratio2[date].float()
                expanded_sum2[i1,grouped_sid2[date]] = grouped_sum2[date].float()
                expanded_txt2_sc[i1, grouped_sid2[date]] = grouped_txt2_sc[date].float()
                last = i1+1


            expanded_tensor2 = expanded_tensor2[:last, :,:,:]
            expanded_label2 = expanded_label2[:last,:,:]
            expanded_txt2 = expanded_txt2[:last,:,:,:]
            expanded_txt2_sc = expanded_txt2_sc[:last,:,:]
            expanded_tensor2t = expanded_tensor2.permute(0,2,1,3)
            pred, test_emb = dtml(expanded_tensor2t.to(device))
            data_test = []
            label_test = []
            test_num = []
            for k in range(len(expanded_label2)):
                data_test.append(pred[k][(expanded_label2[k]!=torch.tensor(2)).squeeze(1).to(device)])
                label_test.append(expanded_label2[k].to(device)[(expanded_label2[k]!=torch.tensor(2)).squeeze(1).to(device)])
                test_num.append(pred[k])
            test_num = torch.cat(test_num, dim=0)
            test_num_count += len(test_num)
            data_test2 = torch.cat(data_test, dim=0)
            label_test2 = torch.cat(label_test, dim=0).squeeze(-1)
            countall2 += len(label_test2)
            predicted1 = (torch.sigmoid(data_test2) > 0.5).int()
            correct4 += (label_test2.to(device) == predicted1.to(device).squeeze(1)).sum().item()
            test_emb = test_emb.permute(1,0,2)
            expanded_label21 = expanded_label2.permute(1,0,2)
            expanded_txt_b = expanded_txt2.permute(1,0,2,3)
            expanded_txt_sc_b = expanded_txt2_sc.permute(1,0,2)
            pred = []
            for o in range(len(expanded_label21)):
                out = context_model(test_emb[o].detach().unsqueeze(0), expanded_txt_b[o].unsqueeze(0), expanded_txt_sc_b[o].unsqueeze(0).to(device))
                label = expanded_label21[o].clone()
                loss = criterion(out[0][(expanded_label21[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].to(device), label.to(device)[(expanded_label21[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                predicted = (torch.sigmoid(out[0][(expanded_label21[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].to(device)) > 0.5).int()
                correct_test += (label.to(device)[(expanded_label21[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].squeeze(1).to(device) == predicted.squeeze(1)).sum().item()
                predict_all.append(predicted.squeeze(1))
                label_all.append(label.to(device)[(expanded_label21[o].to(device)!=torch.tensor(2).to(device)).squeeze(1)].squeeze(1))
                label_temp = label_all
            predict_all = torch.cat(predict_all,dim=0)
            label_all = torch.cat(label_all,dim=0)
            print(correct_eval/countall1,correct_test/countall2,accuracy_score(label_all.detach().cpu().numpy(),predict_all.detach().cpu().numpy()),calculate_mcc(label_all.detach().cpu().numpy(),predict_all.detach().cpu().numpy()))    
            if accuracy_score(label_all_eval.detach().cpu().numpy(),predict_all_eval.detach().cpu().numpy())>best and epoch>0:
                best = accuracy_score(label_all_eval.detach().cpu().numpy(),predict_all_eval.detach().cpu().numpy())
                torch.save(context_model,"/home/zhaokx/stock_text_pred/RQ-stock/checkpoints/stocknet_text/"+"cn2_model_time_text.pth")
                print("***",accuracy_score(label_all_eval.detach().cpu().numpy(),predict_all_eval.detach().cpu().numpy()),accuracy_score(label_all.detach().cpu().numpy(),predict_all.detach().cpu().numpy()))
                best_test_acc = accuracy_score(label_all.detach().cpu().numpy(),predict_all.detach().cpu().numpy())
    

