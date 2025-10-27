import torch
import torch.nn as nn
class CrossAttentionWithEmbeddedReferenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, embedding_dim):
        super(CrossAttentionWithEmbeddedReferenceModel, self).__init__()
        # 特征投影到隐藏空间
        self.proj_query = nn.Linear(input_dim, hidden_dim)
        self.proj_key_value = nn.Linear(input_dim, hidden_dim)
        # 多头跨注意力机制
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.reference_embedding = nn.Embedding(2, hidden_dim//3)  
        self.fusion_fc = nn.Linear(hidden_dim+hidden_dim//3+hidden_dim,5)
        self.fc1 = nn.Linear(8,1)
        self.vqac = VQACircuit()

    def forward(self, x1, x2, true_labels_x2=None, reference_weights_x2=None, pred_res = None,dis=None):
        """
        x1: (batch_size, 1, input_dim) -> 要预测的特征
        x2: (batch_size, 5, input_dim) -> 参考特征
        true_labels_x2: (batch_size, 1) -> 真值标签
        reference_weights_x2: (batch_size, 5) -> 参考标签
        """
        batch_size = x1.size(0)
        x1 = x1.unsqueeze(1)  # (batch_size, 5, input_dim)
        query = self.proj_query(x2)
        key = self.proj_key_value(x1)
        value = key
        attn_output1, attn_weights1 = self.cross_attention(query, key, value)
        if reference_weights_x2 is not None:
            reference_embeds = self.reference_embedding(reference_weights_x2.long())  # (batch_size, 5, embedding_dim)
            attn_output2 = torch.cat([attn_output1,query,reference_embeds],dim=-1)
        combined_weights = torch.softmax(
        attn_weights1, dim=-1)  # (batch_size, 5）
        weighted_sum = torch.sum(attn_output2 * combined_weights, dim=1)  # (batch_size, embed_dim)
        label_temp = self.fusion_fc(weighted_sum)
        # print("1706",pred_res.shape,label_temp.shape)
        enhanced_pred = pred_res.unsqueeze(-1) + self.vqac(torch.cat([pred_res.unsqueeze(1),label_temp],dim=1)).unsqueeze(1)
        return enhanced_pred, None, enhanced_pred,  true_labels_x2.squeeze(-1).long(), label_temp