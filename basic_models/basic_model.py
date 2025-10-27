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
        self.mefi = MambaTimeSeriesModel(8,64)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, x = self.mefi(x)
        out, _ = self.lstm(x)             # out: (batch, seq_len, hidden_size)
        last = out[:, -1, :]              # 取最后一个时间步
        logits = self.fc(last).squeeze(-1)# (batch,)
        return logits

def objective(trial):
    set_seed(43)
    best_eval_acc = 0
    best_test_acc = 0
    torch.cuda.empty_cache()
    try:
        layers = [1,2,3]
        heads = [1,2,3]
        dims = [32,48,64,96,128]
        pos_weis = [0.7,0.8,0.9,1.0]
        layers = trial.suggest_int("layers",1,4)
        heads = trial.suggest_int("heads",1,4)
        dims = trial.suggest_categorical("dims",[32,48,64])
        pos_weis = trial.suggest_categorical("pos_weis",[0.7,0.8,0.9,1.0])
        learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
        mamba_ly = trial.suggest_categorical("mamba_ly",[1,2,3])
        
        gl_wei = trial.suggest_categorical("gl_wei",[0.1,0.2,0.3])
        dtml = DTML_trans(gl_wei, 8, dims, max1+1, heads, layers,mamba_ly).to(device)
        optimizer = optim.Adam(dtml.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([pos_weis]).to(device))
        best = 0
        loss_last = 0
        idx = 600
        best_eval_acc = 0
        best_test_acc = 0
        for epoch in range(num_epochs):
            print(epoch)
            correct2 = 0
            dtml.train()
            correct = 0
            count = 0
            correct = 0
            for j in range(0,len(expanded_tensor),idx):
                optimizer.zero_grad()
                data_train = []
                label_train = []
                dt_train = expanded_tensor[j:j+idx,:,:].to(device)
                # dt_text = expanded_txt[j:j+idx,:,:,:].to(device)
                pred, train_emb = dtml(dt_train)
                dt_label = expanded_label[j:j+idx,:].to(device)
                lb_mask = []
                for i in range(len(dt_label)):
                    data_train.append(pred[i][(dt_label[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                    label_train.append(dt_label[i][(dt_label[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                data_train1 = torch.cat(data_train, dim=0)
                label_train1 = torch.cat(label_train, dim=0).squeeze(-1)
                predicted = (torch.sigmoid(data_train1) > 0.5).int()
                # predicted = torch.argmax(data_train1[(label_train1!=torch.tensor(2).to(device))],dim=1)
                # correct += (label_train1.to(device)[(label_train1!=torch.tensor(2).to(device))] == predicted.to(device)).sum().item()
                # count += len(predicted)
                loss = criterion(data_train1[(label_train1!=torch.tensor(2).to(device))], label_train1.float()[(label_train1!=torch.tensor(2).to(device))])
                loss.backward()
                optimizer.step()
            train_emb1 = train_emb.reshape(-1, train_emb.size(2))  # [251 * 24, 128]
            indices = torch.arange(train_emb1.size(1)).repeat(train_emb1.size(0))  # [251 * 24]
            dtml.eval()
            data_eval = []
            label_eval = []
            
            with torch.no_grad():
                pred, eval_emb = dtml(expanded_tensor1.to(device))
                for i in range(len(expanded_label1)):
                    data_eval.append(pred[i][(expanded_label1[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                    label_eval.append(expanded_label1[i].to(device)[(expanded_label1[i].to(device)!=torch.tensor(2).to(device)).squeeze(1)])
                data_eval1 = torch.cat(data_eval, dim=0).to(device)
                label_eval1 = torch.cat(label_eval, dim=0).squeeze(-1).to(device)
                loss1 = criterion(data_eval1.squeeze(), label_eval1.float())

                eval_emb1 = eval_emb.reshape(-1, eval_emb.size(2))  # [251 * 24, 128]
                indices1 = torch.arange(eval_emb1.size(1)).repeat(eval_emb1.size(0))  # [251 * 24]
                predicted1 = (torch.sigmoid(data_eval1) > 0.5).int()
                correct3 = (label_eval1.to(device) == predicted1.to(device)).sum().item()
                data_test = []
                label_test = []
                pred, test_emb = dtml(expanded_tensor2.to(device))
                for i in range(len(expanded_label2)):
                    data_test.append(pred[i][(expanded_label2[i]!=torch.tensor(2)).squeeze(1).to(device)])
                    label_test.append(expanded_label2[i].to(device)[(expanded_label2[i]!=torch.tensor(2)).squeeze(1).to(device)])
                data_test2 = torch.cat(data_test, dim=0)
                label_test2 = torch.cat(label_test, dim=0).squeeze(-1)
                predicted = (torch.sigmoid(data_test2) > 0.5).int()
                correct2 = (label_test2.to(device) == predicted.to(device)).sum().item()
                test_emb1 = test_emb.reshape(-1, test_emb.size(2))  # [251 * 24, 128]
                if correct3/len(predicted1)>best:
                    best = correct3/len(predicted1)
                    torch.save(dtml,"./../checkpoints/"+dir1+"/"+dir1+"_".join([str(b) for b in [layers,heads,dims,pos_weis,learning_rate,gl_wei,mamba_ly]])+"_model_time_con.pth")
                    best_eval_acc = correct3/len(predicted1)
                    best_test_acc = correct2/len(predicted)
                    print("***",abs(best_eval_acc-best_test_acc),correct2/len(predicted),calculate_mcc(label_test2.detach().cpu().numpy(),predicted.detach().cpu().numpy()),correct3/len(predicted1),calculate_mcc(label_eval1.detach().cpu().numpy(),predicted1.detach().cpu().numpy()))
                loss_last = loss1
        print(best_eval_acc)
    except Exception as e:
        # raise e
        print(e)
        pass

    return abs(best_eval_acc-best_test_acc) if abs(best_eval_acc-best_test_acc)!=0 else 1000


if __name__ == "__main__":
    set_seed(1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dir1 = "CMIN-CN"
    num_epochs = 100
    loaded = np.load('./../datasets/'+dir1+'_data.npz')
    tra_pv = torch.tensor(loaded['tra_pv'])
    train_label = torch.tensor(loaded['tra_gt'])
    tra_wd = torch.tensor(loaded['tra_wd'])
    tra_dt = torch.tensor(loaded['tra_dt'])
    tra_tn = torch.tensor(loaded['tra_co'])
    tra_lb = torch.tensor(loaded['tra_lb'])
    val_pv = torch.tensor(loaded['val_pv'])
    val_label = torch.tensor(loaded['val_gt'])
    val_wd = torch.tensor(loaded['val_wd'])
    val_dt = torch.tensor(loaded['val_dt'])
    val_tn = torch.tensor(loaded['val_co'])
    val_lb = torch.tensor(loaded['val_lb'])
    tes_pv = torch.tensor(loaded['tes_pv'])
    test_label = torch.tensor(loaded['tes_gt'])
    tes_wd = torch.tensor(loaded['tes_wd'])
    tes_dt = torch.tensor(loaded['tes_dt'])
    tes_tn = torch.tensor(loaded['tes_co'])
    tes_lb = torch.tensor(loaded['tes_lb'])


    train_data = torch.tensor(np.concatenate((tra_pv,tra_wd),axis=-1))
    test_data = torch.tensor(np.concatenate((tes_pv,tes_wd),axis=-1))
    val_data = torch.tensor(np.concatenate((val_pv,val_wd),axis=-1))

    batch_size, seq_len, feature_dim = train_data.shape
    print(train_data.shape)
    best = 0 
    acc = 0
    idx = 1000
    dis = 1000
    train_data = torch.tensor(train_data)
    grouped_data = defaultdict(list)
    grouped_label = defaultdict(list)
    grouped_sid = defaultdict(list)
    grouped_lb = defaultdict(list)
    max1 = 0
    for i in range(train_data.shape[0]):
        date = tra_dt[i].item()
        grouped_data[date].append(train_data[i])
        grouped_label[date].append(train_label[i])
        grouped_sid[date].append(tra_tn[i][0].long())
        grouped_lb[date].append(tra_lb[i][0].long())
        if tra_tn[i][0]>max1:
            max1 = tra_tn[i][0].int()
    print(len(grouped_data))
    grouped_data = {k: torch.stack(v) for k, v in grouped_data.items()}
    grouped_label = {k: torch.stack(v) for k, v in grouped_label.items()}
    grouped_sid = {k: torch.stack(v) for k, v in grouped_sid.items()}
    grouped_lb = {k: torch.stack(v) for k, v in grouped_lb.items()}
    # 初始化张量（用 padding 统一维度）
    expanded_tensor = torch.zeros(len(grouped_data), max1+1, 5, 8)
    expanded_label = torch.zeros(len(grouped_data), max1+1,1 )
    # 将数据填充到 expanded_tensor
    numbers0 = {}
    numbers01 = {}
    for i, (date, samples) in enumerate(grouped_data.items()):
        expanded_tensor[i, grouped_sid[date]] = samples.float()
        expanded_label[i,grouped_sid[date]] = grouped_label[date].float()
        numbers0[i] = grouped_sid[date]
        numbers01[i] = (grouped_lb[date] == torch.tensor(1))
        numbers0[i] = numbers0[i][numbers01[i]]
        
    expanded_tensor = expanded_tensor.permute(0,2,1,3)
    val_data = torch.tensor(val_data)
    grouped_data1 = defaultdict(list)
    grouped_label1 = defaultdict(list)
    grouped_sid1 = defaultdict(list)
    grouped_lb1 = defaultdict(list)
    
    for i in range(val_data.shape[0]):
        date = val_dt[i].item()
        grouped_data1[date].append(val_data[i])
        grouped_label1[date].append(val_label[i])
        grouped_sid1[date].append(val_tn[i][0].long())
        grouped_lb1[date].append(val_lb[i][0].long())

    grouped_data1 = {k: torch.stack(v) for k, v in grouped_data1.items()}
    grouped_label1 = {k: torch.stack(v) for k, v in grouped_label1.items()}
    grouped_sid1 = {k: torch.stack(v) for k, v in grouped_sid1.items()}
    grouped_lb1 = {k: torch.stack(v) for k, v in grouped_lb1.items()}

    # 初始化张量（用 padding 统一维度）
    expanded_tensor1 = torch.zeros(len(grouped_data1), max1+1, 5, 8)
    expanded_label1 = torch.zeros(len(grouped_data1), max1+1,1 )
    # 将数据填充到 expanded_tensor
    numbers1 = {}
    numbers11 = {}
    
    for i, (date, samples) in enumerate(grouped_data1.items()):
        expanded_tensor1[i, grouped_sid1[date]] = samples.float()
        expanded_label1[i, grouped_sid1[date]] = grouped_label1[date].float()
        numbers1[i] = grouped_sid1[date]
        numbers11[i] = grouped_lb1[date]
        numbers11[i] = (grouped_lb1[date] == torch.tensor(1))
        numbers1[i] = numbers1[i][numbers11[i]]
    expanded_tensor1 = expanded_tensor1.permute(0,2,1,3)

    test_data = torch.tensor(test_data)
    grouped_data2 = defaultdict(list)
    grouped_label2 = defaultdict(list)
    grouped_sid2 = defaultdict(list)
    grouped_lb2 = defaultdict(list)

    for i in range(test_data.shape[0]):
        date = tes_dt[i].item()
        grouped_data2[date].append(test_data[i])
        grouped_label2[date].append(test_label[i])
        grouped_sid2[date].append(tes_tn[i][0].long())
        grouped_lb2[date].append(tes_lb[i][0].long())
    
    grouped_data2 = {k: torch.stack(v) for k, v in grouped_data2.items()}
    grouped_label2 = {k: torch.stack(v) for k, v in grouped_label2.items()}
    grouped_sid2 = {k: torch.stack(v) for k, v in grouped_sid2.items()}
    grouped_lb2 = {k: torch.stack(v) for k, v in grouped_lb2.items()}
    # 初始化张量（用 padding 统一维度）
    expanded_tensor2 = torch.zeros(len(grouped_data2), max1+1, 5, 8)
    expanded_label2 = torch.zeros(len(grouped_data2), max1+1, 1)
    # 将数据填充到 expanded_tensor
    numbers2 = {}
    numbers21 = {}
    for i, (date, samples) in enumerate(grouped_data2.items()):
        expanded_tensor2[i, grouped_sid2[date]] = samples.float()
        expanded_label2[i, grouped_sid2[date]] = grouped_label2[date].float()
        numbers2[i] = grouped_sid2[date]
        numbers21[i] = grouped_lb2[date]
        numbers21[i] = (grouped_lb2[date] == torch.tensor(1))
        numbers2[i] = numbers2[i][numbers21[i]]
    expanded_tensor2 = expanded_tensor2.permute(0,2,1,3)
    sampler = optuna.samplers.TPESampler(seed=2)
    study = optuna.create_study(direction="minimize", sampler=sampler)  # 最大化目标值
    study.optimize(objective, n_trials=50)  # 运行 50 次试验


# test without dtml
# # 简单的训练/验证/测试 loop（占位 model，二分类 BCEWithLogitsLoss）
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# # 使用文件中已有的 train_data, val_data, test_data 与对应的标签变量
# X_train = train_data.float()
# y_train = train_label.view(-1)     # 保持原始值（可能含 2）
# X_val = val_data.float()
# y_val = val_label.view(-1)
# X_test = test_data.float()
# y_test = test_label.view(-1)

# from torch.utils.data import TensorDataset, DataLoader
# batch_size = 1280
# train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# input_dim = X_train.shape[-1]


# input_size = X_train.shape[-1]
# model = LSTMClassifier(input_size=input_size, hidden_size=128, num_layers=1, dropout=0.0).to(device)

# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=5e-5)

# num_epochs = 40
# best_val_acc = -1.0
# best_state = None

# for epoch in range(1, num_epochs + 1):
#     model.train()
#     train_loss_sum = 0.0
#     train_valid_samples = 0

#     for xb, yb in train_loader:
#         xb = xb.to(device)                            # shape (batch, seq_len, input_size)
#         yb = yb.to(device)

#         mask = (yb != 2)                              # 过滤掉标签为 2 的样本
#         if mask.sum().item() == 0:
#             continue                                  # 该 batch 没有有效样本，跳过

#         logits = model(xb).squeeze(-1)                # (batch,)
#         logits_m = logits[mask]
#         yb_m = yb[mask].float()

#         optimizer.zero_grad()
#         loss = criterion(logits_m, yb_m)
#         loss.backward()
#         optimizer.step()

#         train_loss_sum += loss.item() * mask.sum().item()
#         train_valid_samples += mask.sum().item()

#     train_loss = train_loss_sum / train_valid_samples if train_valid_samples > 0 else 0.0

#     # 验证
#     model.eval()
#     val_loss_sum = 0.0
#     val_valid_samples = 0
#     val_preds = []
#     val_labels = []
#     with torch.no_grad():
#         for xb, yb in val_loader:
#             xb = xb.to(device)
#             yb = yb.to(device)

#             mask = (yb != 2)
#             if mask.sum().item() == 0:
#                 continue

#             logits = model(xb).squeeze(-1)
#             logits_m = logits[mask]
#             yb_m = yb[mask].float()

#             loss = criterion(logits_m, yb_m)
#             val_loss_sum += loss.item() * mask.sum().item()
#             val_valid_samples += mask.sum().item()

#             probs = torch.sigmoid(logits_m)
#             preds = (probs > 0.5).long().cpu().numpy()
#             val_preds.append(preds)
#             val_labels.append(yb_m.cpu().numpy().astype(int))

#     val_loss = val_loss_sum / val_valid_samples if val_valid_samples > 0 else 0.0
#     if val_valid_samples > 0:
#         val_preds = np.concatenate(val_preds, axis=0)
#         val_labels = np.concatenate(val_labels, axis=0)
#         val_acc = (val_preds == val_labels).mean()
#         val_mcc = calculate_mcc(val_labels, val_preds)
#     else:
#         val_acc = 0.0
#         val_mcc = 0.0

#     # 保存最佳模型（按 val_mcc）
#     if val_mcc > best_val_acc:
#         best_val_acc = val_acc
#         best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#         print(best_val_acc)
#     print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_mcc={val_mcc:.4f}")

# # 在测试集上评估（使用 best_state）
# if best_state is not None:
#     model.load_state_dict(best_state)

# model.eval()
# test_loss_sum = 0.0
# test_valid_samples = 0
# test_preds = []
# test_labels = []
# with torch.no_grad():
#     for xb, yb in test_loader:
#         xb = xb.to(device)
#         yb = yb.to(device)

#         mask = (yb != 2)
#         if mask.sum().item() == 0:
#             continue

#         logits = model(xb).squeeze(-1)
#         logits_m = logits[mask]
#         yb_m = yb[mask].float()

#         loss = criterion(logits_m, yb_m)
#         test_loss_sum += loss.item() * mask.sum().item()
#         test_valid_samples += mask.sum().item()

#         probs = torch.sigmoid(logits_m)
#         preds = (probs > 0.5).long().cpu().numpy()
#         test_preds.append(preds)
#         test_labels.append(yb_m.cpu().numpy().astype(int))

# test_loss = test_loss_sum / test_valid_samples if test_valid_samples > 0 else 0.0
# if test_valid_samples > 0:
#     test_preds = np.concatenate(test_preds, axis=0)
#     test_labels = np.concatenate(test_labels, axis=0)
#     test_acc = (test_preds == test_labels).mean()
#     test_mcc = calculate_mcc(test_labels, test_preds)
# else:
#     test_acc = 0.0
#     test_mcc = 0.0

# print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f} mcc={test_mcc:.4f}")