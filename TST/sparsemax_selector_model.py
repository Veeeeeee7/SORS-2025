import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadedAttention(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_attention = d_model // num_heads
        self.dropout = dropout

    def forward(self, query, key, value):
        n, d_model = query.shape
        m, d_model = key.shape

        Qh = query.view(n, self.num_heads, self.d_attention).permute(1, 0, 2)
        Kh = key.view(m, self.num_heads, self.d_attention).permute(1, 0, 2)
        Vh = value.view(m, self.num_heads, self.d_attention).permute(1, 0, 2)

        attn_scores = torch.matmul(Qh, Kh.transpose(1, 2)) / (self.d_attention ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        context_per_head = torch.matmul(attn_weights, Vh)

        context = context_per_head.permute(1, 0, 2).contiguous().view(n, d_model)

        return context, attn_scores

class TimeSeriesTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout):
        super().__init__()

        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)

        # self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
        #                                             num_heads=num_heads,
        #                                             dropout=dropout,
        #                                             bias=True,
        #                                             batch_first=True)

        self.self_attn = MultiheadedAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm_attn = nn.BatchNorm1d(d_model)
        self.norm_ffn = nn.BatchNorm1d(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, X):
        # X: (num_sensed, seq_len, d_model)
        num_sensed, seq_len, d_model = X.shape

        # Compute Query, Key, Value Embeddings
        query = X.view(num_sensed*seq_len, d_model)
        query = self.query_embedding(query)
        key = X.view(num_sensed*seq_len, d_model)
        key = self.key_embedding(key)
        value = X.view(num_sensed*seq_len, d_model)
        value = self.value_embedding(value)

        # Attention Mechanism
        # attn_output, attn_scores = self.self_attn(query, key, value, need_weights=True, average_attn_weights=True)
        attn_output, attn_scores = self.self_attn(query, key, value)
        attn_output = attn_output.view(num_sensed, seq_len, d_model)

        # Dropout + Residual Connection
        dropout_attn = self.dropout_attn(attn_output)
        residual_attn = X + dropout_attn

        # Batch Norm
        residual_attn = residual_attn.view(num_sensed*seq_len, d_model)
        residual_attn = residual_attn.unsqueeze(0)
        residual_attn = residual_attn.permute(0, 2, 1)
        norm_attn = self.norm_attn(residual_attn)
        norm_attn = norm_attn.permute(0, 2, 1)
        norm_attn = norm_attn.squeeze(0)
        norm_attn = norm_attn.view(num_sensed, seq_len, d_model)
        norm_attn = norm_attn.view(num_sensed*seq_len, d_model)

        # Feed Forward Network
        ffn_output = self.ffn(norm_attn)
        ffn_output = ffn_output.view(num_sensed, seq_len, d_model)

        # Dropout + Residual Connection
        dropout_ffn = self.dropout_ffn(ffn_output)
        norm_attn = norm_attn.view(num_sensed, seq_len, d_model)
        residual_ffn = norm_attn + dropout_ffn

        # Batch Norm
        residual_ffn = residual_ffn.view(num_sensed*seq_len, d_model)
        residual_ffn = residual_ffn.unsqueeze(0)
        residual_ffn = residual_ffn.permute(0, 2, 1)
        norm_ffn = self.norm_ffn(residual_ffn)
        norm_ffn = norm_ffn.permute(0, 2, 1)
        norm_ffn = norm_ffn.squeeze(0)
        norm_ffn = norm_ffn.view(num_sensed, seq_len, d_model)
        return norm_ffn, attn_scores
    
class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 num_layers,
                 dropout):
        super().__init__()

        self.layers = nn.ModuleList([
            TimeSeriesTransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, X):
        output = X
        for layer in self.layers:
            output, attn_scores = layer(output)
        return output, attn_scores

class RandomSelector(nn.Module):
    def __init__(self,
                 top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, X):
        # X: (num_sensed, seq_len, num_variables)
        indices = torch.randperm(X.shape[0], device=X.device)[:self.top_k]
        return indices
    
class SparsemaxSelector(nn.Module):
    def __init__(self, num_sensors: int, top_k: int, eps: float = 1e-6):
        super().__init__()
        self.num_sensors = num_sensors
        self.top_k = top_k
        self.eps = eps
        # we can't learn weights because the outputs aren't "used" in the current training loop and gradients wont propagate to these weights
        # we will use the attention scores outputed from the encoder as the weights
        # self.weights = nn.Parameter(torch.randn(num_sensors))  # Learnable scores

    # def sparsemax(self, z: torch.Tensor) -> torch.Tensor:
    #     # Sparsemax implementation (simplified)
    #     z_sorted, _ = torch.sort(z, descending=True)
    #     cumsum = torch.cumsum(z_sorted, dim=0)
    #     k = torch.arange(1, len(z) + 1, device=z.device)
    #     threshold = (z_sorted - (cumsum - 1) / k) > 0
    #     k_z = torch.sum(threshold.float())
    #     tau_z = (cumsum[k_z - 1] - 1) / k_z
    #     return torch.clamp(z - tau_z, min=0)

    # errors in original sparsemax, using chatgpt fixes below
    # @Carson can you check for mathematical accuracy
    def sparsemax(self, z: torch.Tensor) -> torch.Tensor:
        # 1) sort z descending
        z_sorted, _ = torch.sort(z, descending=True)
        # 2) cumulative sums
        cumsum = torch.cumsum(z_sorted, dim=0)
        # 3) arange k = 1…n (float for the div)
        k = torch.arange(1, z.numel()+1, device=z.device, dtype=z.dtype)
        # 4) find how many entries satisfy the threshold
        threshold = (z_sorted - (cumsum - 1) / k) > 0
        k_z = int(threshold.sum().item())    # now a Python int
        # 5) compute τ
        tau_z = (cumsum[k_z - 1] - 1) / k_z
        # 6) clamp
        return torch.clamp(z - tau_z, min=0.0)


    def forward(self, scores: torch.Tensor, beta: float = 1.0):
        # scores = self.weights * X.mean(dim=1)  # Combine learned weights + input
        probs = self.sparsemax(scores / beta)  # Apply sparsemax
        indices = torch.topk(probs, self.top_k).indices  # Select top-k indices
        return indices

class StatisticalGumbelTopKSelector(nn.Module):
    def __init__(self,
                 top_k,
                 eps=1e-6,
                 ):
        super().__init__()
        self.top_k = top_k
        self.eps = eps

    def sample_gumbel(self, shape, device):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + self.eps) + self.eps)

    def forward(self, X, beta):
        num_sensors = X.size(0)
        with torch.no_grad():
            scores = X.mean(dim=1)
            gumbel = self.sample_gumbel((num_sensors,), X.device)
            noisy_scores = (scores + gumbel) / beta
            topk_inds = torch.topk(noisy_scores, self.top_k).indices

        return topk_inds

class TimeSeriesTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout):
        super().__init__()

        # self.cross_attn = nn.MultiheadAttention(embed_dim=d_model,
        #                                              num_heads=num_heads,
        #                                              dropout=dropout,
        #                                              bias=True,
        #                                              batch_first=True)
        self.cross_attn = MultiheadedAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        
        # self.norm_cross_attn = nn.LayerNorm(d_model)
        # self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_cross_attn = nn.BatchNorm1d(d_model)
        self.norm_ffn = nn.BatchNorm1d(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout_cross_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        # Q: (num_unsensed, seq_len, d_model)
        # K, V: (num_unsensed, num_sensed, d_model)
        num_unsensed, seq_len, d_model = Q.shape
        _, num_sensed, _ = K.shape

        # Reshape Q, K, V for Cross Attention
        Q = Q.view(num_unsensed*seq_len, d_model)
        K = K.contiguous().view(num_unsensed*num_sensed, d_model)
        V = V.contiguous().view(num_unsensed*num_sensed, d_model)

        # Cross Attention Mechanism
        # cross_attn_output, cross_attn_scores = self.cross_attn(Q, K, V, need_weights=True, average_attn_weights=True)
        cross_attn_output, cross_attn_scores = self.cross_attn(Q, K, V)
        cross_attn_output = cross_attn_output.view(num_unsensed, seq_len, d_model)

        # Dropout + Residual Connection
        dropout_attn = self.dropout_cross_attn(cross_attn_output)
        Q = Q.view(num_unsensed, seq_len, d_model)
        residual_attn = Q + dropout_attn

        # Batch Norm
        residual_attn = residual_attn.view(num_unsensed*seq_len, d_model)
        residual_attn = residual_attn.unsqueeze(0)
        residual_attn = residual_attn.permute(0, 2, 1)
        norm_attn = self.norm_cross_attn(residual_attn)
        norm_attn = norm_attn.permute(0, 2, 1)
        norm_attn = norm_attn.squeeze(0)
        norm_attn = norm_attn.view(num_unsensed, seq_len, d_model)
        norm_attn = norm_attn.view(num_unsensed*seq_len, d_model)

        # Feed Forward Network
        ffn_output = self.ffn(norm_attn)
        ffn_output = ffn_output.view(num_unsensed, seq_len, d_model)

        # Dropout + Residual Connection
        dropout_ffn = self.dropout_ffn(ffn_output)
        norm_attn = norm_attn.view(num_unsensed, seq_len, d_model)
        residual_ffn = norm_attn + dropout_ffn

        # Batch Norm
        residual_ffn = residual_ffn.view(num_unsensed*seq_len, d_model)
        residual_ffn = residual_ffn.unsqueeze(0)
        residual_ffn = residual_ffn.permute(0, 2, 1)
        norm_ffn = self.norm_ffn(residual_ffn)
        norm_ffn = norm_ffn.permute(0, 2, 1)
        norm_ffn = norm_ffn.squeeze(0)
        norm_ffn = norm_ffn.view(num_unsensed, seq_len, d_model)
        return norm_ffn
    
class TimeSeriesTransformerDecoder(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 num_layers,
                 dropout):
        super().__init__()

        self.layers = nn.ModuleList([
            TimeSeriesTransformerDecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, Q, K, V):
        output = Q
        for layer in self.layers:
            output = layer(output, K, V)
        return output

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 num_variables: int = 6,
                 num_static: int = 2,
                 seq_len: int = 24,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 num_sensors: int = 100,
                 top_k: int = 50,
                 dropout: float = 0.1,
                 eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.d_attention = d_model // num_heads
        self.num_variables = num_variables
        self.num_static = num_static
        self.seq_len = seq_len
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_sensors = num_sensors
        self.top_k = top_k
        self.dropout = dropout
        self.eps = eps

        self.embedding_projection = nn.Linear(num_variables, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(num_static, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.positional_encoder = nn.Parameter(torch.randn(seq_len, d_model))

        self.encoder = TimeSeriesTransformerEncoder(d_model=d_model,
                                                    num_heads=num_heads,
                                                    d_ff=d_ff,
                                                    num_layers=num_encoder_layers,
                                                    dropout=dropout)
        self.encoder_output_projection = nn.Linear(seq_len, 1)
        self.selector = SparsemaxSelector(num_sensors=num_sensors, top_k=top_k, eps=eps)
        self.decoder = TimeSeriesTransformerDecoder(d_model=d_model,
                                                    num_heads=num_heads,
                                                    d_ff=d_ff,
                                                    num_layers=num_decoder_layers,
                                                    dropout=dropout)
        
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, X, static, unsensed_static, beta):
        """
        Embedding
        """
        # X: (num_sensed, seq_len, num_variables)
        num_sensed, _, _ = X.shape 

        X = X.view(num_sensed*self.seq_len, self.num_variables)
        X_embedding = self.embedding_projection(X)
        X_embedding = X_embedding.view(num_sensed, self.seq_len, self.d_model)

        # static: (num_sensed, num_static)
        static_embedding = self.mlp(static)
        static_embedding = static_embedding.unsqueeze(1)
        static_embedding_broadcasted = static_embedding.expand(num_sensed, self.seq_len, self.d_model)

        # positional_encoder: (seq_len, d_model)
        positional_encoding = self.positional_encoder.unsqueeze(0)
        positional_encoding = positional_encoding.expand(num_sensed, self.seq_len, self.d_model)

        embedding = X_embedding + static_embedding_broadcasted + positional_encoding

        """
        Encoder
        """
        encoder_output, attn_scores = self.encoder(embedding)
        encoder_output = encoder_output.contiguous().view(num_sensed*self.d_model, self.seq_len)
        encoder_output = self.encoder_output_projection(encoder_output)
        encoder_output_pooled = encoder_output.view(num_sensed, self.d_model)
        # encoder_output_pooled = encoder_output.mean(dim=1)

        """
        Selector
        """
        selector_input = attn_scores.mean(dim=0) # average attn scores across heads
        selector_input = selector_input.mean(dim=0) # since self attention --> (n,n) scores, average down to (n,)
        selector_input = selector_input.view(num_sensed, self.seq_len) # bring back to (num_sensed, seq_len) from attention shape
        selector_input = selector_input.mean(dim=1) # average across seq_len to get (num_sensed,)

        selected_indices = self.selector(selector_input, beta=beta)

        """
        Decoder
        unsensed_static.shape: (num_unsensed, num_static)
        """
        # unsensed_static: (num_unsensed, num_static)
        num_unsensed, _ = unsensed_static.shape

        unsensed_static_embedding = self.mlp(unsensed_static)
        unsensed_static_embedding = unsensed_static_embedding.unsqueeze(1)
        unsensed_static_embedding_broadcasted = unsensed_static_embedding.expand(num_unsensed, self.seq_len, self.d_model)

        positional_encoding = self.positional_encoder.unsqueeze(0)
        positional_encoding = positional_encoding.expand(num_unsensed, self.seq_len, self.d_model)

        unsensed_embedding = unsensed_static_embedding_broadcasted + positional_encoding

        query = unsensed_embedding
        key = encoder_output_pooled.unsqueeze(0)
        key = key.expand(num_unsensed, num_sensed, self.d_model)
        value = key
        decoder_output = self.decoder(query, key, value)

        output = decoder_output.view(num_unsensed*self.seq_len, self.d_model)
        output = self.output_projection(output)
        output = output.view(num_unsensed, self.seq_len)

        return output, selected_indices