import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class NewsDecoder(nn.Module):
    def __init__(self, word_emb_dim, num_filters, window_size, category_emb_dim, num_embeddings, num_categories):
        super(NewsDecoder, self).__init__()
        self.word_embedding = nn.Embedding(num_embeddings, word_emb_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, category_emb_dim)
        self.conv = nn.Conv2d(1, num_filters, (window_size, word_emb_dim))
        self.pool = nn.MaxPool2d((2, 1))
        
        d_model = num_filters * 2 + category_emb_dim
        assert d_model % 2 == 0, "d_model must be divisible by num_heads (2)"

        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=2)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)

    def forward(self, title, body, category, memory):
        title_emb = self.word_embedding(title).unsqueeze(1)  # (batch_size, 1, title_length, word_emb_dim)
        body_emb = self.word_embedding(body).unsqueeze(1)  # (batch_size, 1, body_length, word_emb_dim)
        category_emb = self.category_embedding(category)  # (batch_size, category_emb_dim)

        title_features = self.conv(title_emb)  # (batch_size, num_filters, conv_out_height, 1)
        body_features = self.conv(body_emb)  # (batch_size, num_filters, conv_out_height, 1)

        title_features = self.pool(title_features).squeeze(3)  # (batch_size, num_filters, conv_out_height//2)
        body_features = self.pool(body_features).squeeze(3)  # (batch_size, num_filters, conv_out_height//2)

        title_features = nn.functional.relu(title_features).mean(dim=2)  # (batch_size, num_filters)
        body_features = nn.functional.relu(body_features).mean(dim=2)  # (batch_size, num_filters)

        features = torch.cat([title_features, body_features, category_emb], dim=1)  # (batch_size, num_filters*2 + category_emb_dim)

        features = features.unsqueeze(1)  # (batch_size, 1, num_filters*2 + category_emb_dim)
        output = self.transformer_decoder(features, memory)
        return output.squeeze(1)
