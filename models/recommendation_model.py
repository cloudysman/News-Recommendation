import torch
import torch.nn as nn
import torch.nn.functional as F
from models.news_decoder import NewsDecoder
from models.user_decoder import UserDecoder 
from models.click_predictor import ClickPredictor

class NewsRecommendationModel(nn.Module):
    def __init__(self, word_emb_dim, num_filters, window_size, category_emb_dim, num_embeddings, num_categories):
        super(NewsRecommendationModel, self).__init__()
        self.news_encoder = NewsDecoder(word_emb_dim, num_filters, window_size, category_emb_dim, num_embeddings, num_categories)
        self.user_encoder = UserDecoder(word_emb_dim, num_filters, window_size, category_emb_dim, num_embeddings, num_categories)
        self.click_predictor = ClickPredictor(num_filters * 2 + category_emb_dim)

    def forward(self, title, body, category, clicked_news):
        candidate_news_features = self.news_encoder(title, body, category, None)
        clicked_news_features = [self.user_encoder(c_title, c_body, c_category, candidate_news_features) for c_title, c_body, c_category in clicked_news]

        clicked_news_features = torch.stack(clicked_news_features, dim=1).mean(dim=1)  

        combined_features = torch.cat([candidate_news_features, clicked_news_features], dim=1)
        click_prob = self.click_predictor(combined_features)
        return click_prob

    def get_news_embedding(self, title, body, category, memory):
        return self.news_encoder(title, body, category, memory)
