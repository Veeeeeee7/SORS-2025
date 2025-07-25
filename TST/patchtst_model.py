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

        return context, attn_weights

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
        return norm_ffn
    
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
            output = layer(output)
        return output

class RandomSelector(nn.Module):
    def __init__(self,
                 top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, X):
        # X: (num_sensed, seq_len, num_variables)
        indices = torch.randperm(X.shape[0], device=X.device)[:self.top_k]
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

class AlphaGumbelTopkSelector(nn.Module):
    def __init__(self, num_sensors: int, top_k: int, eps: float = 1e-6):
        super().__init__()
        self.num_sensors = num_sensors
        self.top_k = top_k
        self.eps = eps

        self.alpha = nn.Parameter(torch.rand(num_sensors, top_k) + eps)

    def sample_gumbel(self, shape, device):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + self.eps) + self.eps)

    def forward(self, X, beta):
        log_alpha = torch.log(F.softplus(self.alpha, beta=50.0) + self.eps)
        gumbel = self.sample_gumbel((self.num_sensors, self.top_k), X.device)
        noisy_scores = (log_alpha + gumbel) / beta
        W = torch.softmax(noisy_scores, dim=0)
        Z = torch.matmul(W.transpose(1, 0), X)

        p = self.alpha / (torch.sum(self.alpha, dim=0) + self.eps)
        p_t = p.t()

        # remove negative values
        p_t = F.relu(p_t)

        # renormalize for a probability distribution
        row_sums = p_t.sum(dim=-1, keepdim=True)
        p_t = p_t / (row_sums + self.eps)

        indices = torch.multinomial(p_t, num_samples=1).squeeze(-1)

        return Z, indices, p

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
                 eps: float = 1e-6,
                 patch_len: int = 1):
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
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.k_per_patch = top_k // self.num_patches

        self.embedding_projection = nn.Linear(num_variables, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(num_static, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.positional_encoder = nn.Parameter(torch.randn(seq_len, d_model))

        self.patch_embedding = nn.Conv1d(
            in_channels = d_model,
            out_channels = d_model,
            kernel_size = patch_len,
            stride = patch_len,
            bias = True
        )

        self.encoder = TimeSeriesTransformerEncoder(d_model=d_model,
                                                    num_heads=num_heads,
                                                    d_ff=d_ff,
                                                    num_layers=num_encoder_layers,
                                                    dropout=dropout)
        self.encoder_output_projection = nn.Linear(seq_len, 1)
        self.patch_selectors = nn.ModuleList([
            AlphaGumbelTopkSelector(num_sensors, self.k_per_patch, eps)
            for _ in range(self.num_patches)
        ])
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
        Patching
        """
        emb = embedding.permute(0, 2, 1)  # (num_sensed, d_model, seq_len)
        emb = self.patch_embedding(emb)  # (num_sensed, d_model, num_patches)
        emb = emb.permute(0, 2, 1)  # (num_sensed, num_patches, d_model)


        """
        Encoder
        """
        encoder_output = self.encoder(emb.contiguous()) # (num_sensed, num_patches, d_model)
        # selector_input = encoder_output.mean(dim=2)  # (num_sensed, num_patches)
        selector_input = encoder_output

        """
        Selector
        """
        masked_encoder_embeddings = torch.zeros(self.k_per_patch, self.num_patches, self.d_model, device=encoder_output.device)
        selected_sensors = torch.zeros(self.num_patches, self.k_per_patch, device=encoder_output.device, dtype=torch.int)
        for patch in range(self.num_patches):
            selector = self.patch_selectors[patch]
            selector_input_patch = selector_input[:, patch]
            patch_masked_encoder_embeddings, patch_selected_sensors, _ = selector(selector_input_patch, beta)
            masked_encoder_embeddings[:, patch, :] = patch_masked_encoder_embeddings
            selected_sensors[patch, :] = patch_selected_sensors
        
        selected_sensors = selected_sensors.flatten()

        if self.training:
            all_ids = torch.arange(num_sensed, device=masked_encoder_embeddings.device)
            mask = torch.ones(num_sensed, dtype=torch.bool, device=masked_encoder_embeddings.device)
            mask[selected_sensors] = False

            unselected_indices = all_ids[mask]

            """
            Decoder for unselected_indices
            unselected_indices.shape: (num_unselected, d_model)
            """
            num_unselected = unselected_indices.shape[0]
            # print(f"unselected_indices.shape: {unselected_indices.shape}")
            # print(f"unselected_indices.min: {unselected_indices.min().item()}")
            # print(f"unselected_indices.max: {unselected_indices.max().item()}")
            # print(f"static.shape: {static.shape}")
            unselected_static = static[unselected_indices, :]
            unselected_static_embedding = self.mlp(unselected_static)
            unselected_static_embedding = unselected_static_embedding.unsqueeze(1)
            unselected_static_embedding_broadcasted = unselected_static_embedding.expand(num_unselected, self.seq_len, self.d_model)
            selector_positional_encoding = self.positional_encoder.unsqueeze(0)
            selector_positional_encoding = selector_positional_encoding.expand(num_unselected, self.seq_len, self.d_model)
            unselected_embedding = unselected_static_embedding_broadcasted + selector_positional_encoding

            selector_output = torch.zeros(num_unselected, self.seq_len, device=masked_encoder_embeddings.device)
            for patch in range(self.num_patches):
                patch_selector_query = unselected_embedding[:, patch*self.patch_len:(patch+1)*self.patch_len, :].contiguous()
                patch_selector_key = masked_encoder_embeddings[:, patch, :].unsqueeze(0)
                patch_selector_key = patch_selector_key.expand(num_unselected, self.k_per_patch, self.d_model).contiguous()
                patch_selector_value = patch_selector_key
                patch_selector_decoder_output = self.decoder(patch_selector_query, patch_selector_key, patch_selector_value)
                patch_selector_output = patch_selector_decoder_output.view(num_unselected*self.patch_len, self.d_model)
                patch_selector_output = self.output_projection(patch_selector_output)
                patch_selector_output = patch_selector_output.view(num_unselected, self.patch_len)
                selector_output[:, patch*self.patch_len:(patch+1)*self.patch_len] = patch_selector_output

        else:
            selector_output = None
            p = None

        """
        Decoder for unsensed_indices 
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

        output = torch.zeros(num_unsensed, self.seq_len, device=masked_encoder_embeddings.device)
        for patch in range(self.num_patches):
            patch_selector_query = unsensed_embedding[:, patch*self.patch_len:(patch+1)*self.patch_len, :].contiguous()
            patch_selector_key = encoder_output[:, patch, :].unsqueeze(0)
            patch_selector_key = patch_selector_key.expand(num_unsensed, self.num_sensors, self.d_model).contiguous()
            patch_selector_value = patch_selector_key
            patch_decoder_output = self.decoder(patch_selector_query, patch_selector_key, patch_selector_value)
            patch_output = patch_decoder_output.view(num_unsensed*self.patch_len, self.d_model)
            patch_output = self.output_projection(patch_output)
            patch_output = patch_output.view(num_unsensed, self.patch_len)
            output[:, patch*self.patch_len:(patch+1)*self.patch_len] = patch_output

        return selector_output, output, selected_sensors, _
