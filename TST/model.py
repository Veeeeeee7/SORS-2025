import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(embed_dim=d_model,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    bias=True,
                                                    batch_first=False)

        # WILL CHANGE TO BATCH NORM
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, X):
        attention_output, attention_scores = self.self_attention(query=X, key=X, value=X)
        dropout1 = self.dropout_attention(attention_output)
        residual1 = X + dropout1
        norm1 = self.norm1(residual1)

        ffn_output = self.ffn(norm1)
        dropout2 = self.dropout_ffn(ffn_output)
        residual2 = norm1 + dropout2
        norm2 = self.norm2(residual2)
        return norm2
    
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

class Selector(nn.Module):
    def __init__(self,
                 top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, X):
        indices = torch.randperm(X.shape[0])[:self.top_k]
        return indices

class TimeSeriesTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(embed_dim=d_model,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    bias=True,
                                                    batch_first=False)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model,
                                                     num_heads=num_heads,
                                                     dropout=dropout,
                                                     bias=True,
                                                     batch_first=False)
        
        # WILL CHANGE TO BATCH NORM
        self.norm_self_attention = nn.LayerNorm(d_model)
        self.norm_cross_attention = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout_self_attention = nn.Dropout(dropout)
        self.dropout_cross_attention = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        self_attention_output, self_attention_scores = self.self_attention(query=Q, key=Q, value=Q)
        dropout1 = self.dropout_self_attention(self_attention_output)
        residual1 = Q + dropout1
        norm1 = self.norm_self_attention(residual1)

        cross_attention_output, cross_attention_scores = self.cross_attention(query=norm1, key=K, value=V)
        dropout2 = self.dropout_cross_attention(cross_attention_output)
        residual2 = norm1 + dropout2
        norm2 = self.norm_cross_attention(residual2)

        ffn_output = self.ffn(norm2)
        dropout3 = self.dropout_ffn(ffn_output)
        residual3 = norm2 + dropout3
        norm3 = self.norm_ffn(residual3)

        return norm3
    
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
                 d_model: int = 256,
                 num_variables: int = 6,
                 num_static: int = 2,
                 seq_len: int = 24,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 num_heads: int = 8,
                 d_ff: int = 1024,
                 top_k: int = 50,
                 dropout: float = 0.1):
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
        self.top_k = top_k
        self.dropout = dropout

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
        self.selector = Selector(top_k=top_k)
        self.decoder = TimeSeriesTransformerDecoder(d_model=d_model,
                                                    num_heads=num_heads,
                                                    d_ff=d_ff,
                                                    num_layers=num_decoder_layers,
                                                    dropout=dropout)
        
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, X, static, unsensed_static):
        """
        Embedding
        X.shape: (num_sensors, seq_len, num_variables)
        static.shape: (num_sensors, num_static)
        """
        num_sensors, _, _ = X.shape

        X_embedding = self.embedding_projection(X.view(num_sensors*self.seq_len, self.num_variables))
        X_embedding = X_embedding.view(num_sensors, self.seq_len, self.d_model)

        static_embedding = self.mlp(static)
        static_embedding_broadcasted = static_embedding.unsqueeze(1).expand(num_sensors, self.seq_len, self.d_model)

        positional_encoding = self.positional_encoder.unsqueeze(0)
        positional_encoding = positional_encoding.expand(num_sensors, self.seq_len, self.d_model)

        embedding = X_embedding + static_embedding_broadcasted + positional_encoding
        embedding = embedding.permute(1, 0, 2)

        """
        Encoder
        """
        encoder_output = self.encoder(embedding)
        encoder_output = encoder_output.permute(1, 0, 2)
        encoder_output_pooled = encoder_output.mean(dim=1)

        """
        Selector
        """
        selected_indices = self.selector(X)

        """
        Decoder
        unsensed_static.shape: (num_unsensed, num_static)
        """
        num_unsensed, _ = unsensed_static.shape
        unsensed_static_embedding = self.mlp(unsensed_static)
        unsensed_static_embedding_broadcasted = unsensed_static_embedding.unsqueeze(1).expand(num_unsensed, self.seq_len, self.d_model)

        positional_encoding = self.positional_encoder.unsqueeze(0)
        positional_encoding = positional_encoding.expand(num_unsensed, self.seq_len, self.d_model)

        unsensed_embedding = unsensed_static_embedding_broadcasted + positional_encoding
        unsensed_embedding = unsensed_embedding.permute(1, 0, 2)

        query = unsensed_embedding
        key = encoder_output_pooled.unsqueeze(1).expand(num_sensors, num_unsensed, self.d_model)
        value = key
        decoder_output = self.decoder(query, key, value)

        decoder_output = decoder_output.permute(1, 0, 2)
        output = self.output_projection(decoder_output.reshape(num_unsensed*self.seq_len, self.d_model))
        output = output.view(num_unsensed, self.seq_len)

        return output, selected_indices