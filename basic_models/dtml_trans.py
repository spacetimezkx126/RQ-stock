import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mefi import MambaTimeSeriesModel

class ContextNormalization(nn.Module):
    """
    According to Eq. 3 from [1]
    """
    def __init__(self, num_stocks, hidden_dim):
        super(ContextNormalization, self).__init__()
        # Gamma and beta have the same size as the input h_{u i}
        self.gamma = nn.Parameter(torch.ones(num_stocks, hidden_dim))
        self.beta = nn.Parameter(torch.zeros(num_stocks, hidden_dim))

    def forward(self, x):
        # x: [batch_size, num_stocks, hidden_dim]
        # Calculate the mean and standard deviation across all stocks and the entire hidden space
        # for each batch element separately.
        # This aggregates statistics across the 'num_stocks' and 'hidden_dim' dimensions.
        mean = x.mean(dim=[1, 2], keepdim=True)  # Taking mean across stocks and hidden_dim
        std = x.std(dim=[1, 2], keepdim=True)    # Standard deviation across the same dimensions
        
        # Normalize the input x using the calculated mean and std.
        normalized_x = (x - mean) / (std + 1e-9)

        return self.gamma * normalized_x + self.beta


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, lstm_out):
        # lstm_out shape: [batch_size * num_stocks, seq_len, hidden_dim]
        # Extract the last hidden state as the query vector
        h_T = lstm_out[:, -1, :].unsqueeze(2)  # Shape: [batch_size * num_stocks, hidden_dim, 1]
        
        # Compute attention scores
        # Here, we perform batch matrix multiplication between lstm_out and h_T for each sequence
        # and apply softmax to get attention weights
        attention_scores = torch.bmm(lstm_out, h_T).squeeze(2)  # [batch_size * num_stocks, seq_len]
        alpha_i = F.softmax(attention_scores, dim=1).unsqueeze(2)  # [batch_size * num_stocks, seq_len, 1]
        
        # Compute context vector as a weighted sum of LSTM outputs
        context_vector = torch.sum(alpha_i * lstm_out, dim=1)  # [batch_size * num_stocks, hidden_dim]
        
        return context_vector
    
class AttentiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_stocks, num_layers=1,mamba_ly=2):
        super(AttentiveLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_transform = nn.Linear(input_dim, hidden_dim)
        # print("57**",hidden_dim,num_layers)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)  
        self.mamba_ct = MambaTimeSeriesModel(hidden_dim, 5, mamba_ly, False)
        self.context_norm = ContextNormalization(num_stocks, hidden_dim*2)

    def forward(self, x):
        batch_size, seq_len, num_stocks, input_dim = x.size()
        x_reshaped = x.reshape(batch_size * num_stocks, seq_len, input_dim)
        transformed_features = torch.tanh(self.feature_transform(x_reshaped))
        lstm_out, _ = self.lstm(transformed_features)
        pred, mamba_out = self.mamba_ct(transformed_features)
        lstm_out1 = torch.cat([lstm_out, mamba_out],dim=-1)
        # Apply attention mechanism
        context_vector = self.attention(lstm_out1)
        
        # Reshape context_vector back to the original batch and stocks structure
        context_vector = context_vector.view(batch_size, num_stocks, self.hidden_dim*2)

        # Apply context normalization
        normalized_context_vectors = self.context_norm(context_vector)

        return normalized_context_vectors


class DataAxisSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(DataAxisSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == self.hidden_dim, "hidden_dim must be divisible by num_heads"

        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        num_stocks, batch_size, _ = x.size()
        
        # Eq. (6)
        Q = self.Wq(x).view(num_stocks, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        K = self.Wk(x).view(num_stocks, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        V = self.Wv(x).view(num_stocks, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        
        # Calculating attention scores
        # Eq. (7)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim ** 0.5
        S = F.softmax(energy, dim=-1)

        out = torch.matmul(S, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, num_stocks, self.hidden_dim)

        # MLP is applied to the sum of the original input and the aggregated attention output
        # Eq. (8) from [1]
        out_mlp = self.mlp(x.reshape(batch_size, num_stocks, self.hidden_dim) + out)
        out_final = torch.tanh(x.reshape(batch_size, num_stocks, self.hidden_dim) + out + out_mlp) # [batch_size, num_stocks, hidden_dim]

        return out_final


class DTML_trans(nn.Module):
    def __init__(self, beta_hyp,  input_dim, hidden_dim, num_stocks, num_heads, num_layers, mamba_ly):
        super(DTML_trans, self).__init__()
        self.beta_hyp = beta_hyp  # Hyperparameter determining the weight of the global market context
        # self.global_context_index = global_context_index  # Index of the global market context in the dataset
        
        # Updated LSTM model with normalization and attention mechanism
        self.stock_lstm_model = AttentiveLSTM(input_dim, hidden_dim, num_stocks, num_layers, mamba_ly)
        
        # Direct use of DataAxisSelfAttention
        self.self_attention = DataAxisSelfAttention(hidden_dim*2, num_heads)  # Assuming a single head of attention
        self.final_linear = nn.Linear(hidden_dim*2, 1)  # Linear layer for prediction

    def forward(self, x):
        normalized_context_vectors = self.stock_lstm_model(x) # (batch_size, seq_len, num_stocks, input_dim) -> (batch_size, num_stocks, hidden_dim)
        
        # Extract the global market context for the global_context_index
        # global_market_context = normalized_context_vectors[:, self.global_context_index, :].unsqueeze(1)
        global_market_context = torch.mean(normalized_context_vectors,dim=1).unsqueeze(1)
        # Calculate multi-level contexts for each stock (Eq. 4)
        # h^m_u = h^c_u + beta * h^i
        multi_level_contexts = normalized_context_vectors + self.beta_hyp * global_market_context
        
        # Remove the global market context 
        # multi_level_contexts = torch.cat([multi_level_contexts[:, :self.global_context_index, :], multi_level_contexts[:, self.global_context_index+1:, :]], dim=1)
        
        # Apply self-attention mechanism and non-linear transformation
        multi_level_contexts = multi_level_contexts.permute(1, 0, 2)  # Preparing data for self-attention
        attention_output = self.self_attention(multi_level_contexts) # [batch_size, num_stocks, hidden_dim]
        
        # Apply the linear layer to obtain final predictions
        final_predictions = self.final_linear(attention_output).squeeze(-1)
        
        return final_predictions, attention_output