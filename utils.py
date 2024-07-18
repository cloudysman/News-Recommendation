import pandas as pd
import torch
from torch.utils.data import Dataset

class MINDDataset(Dataset):
    def __init__(self, behaviors_path, news_path, word_dict, category_dict, mode='train'):
        self.behaviors = pd.read_csv(behaviors_path, sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'clicked_news', 'impressions'])
        self.news = pd.read_csv(news_path, sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
        
        self.word_dict = word_dict
        self.category_dict = category_dict
        self.mode = mode


        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna('')
        self.behaviors['impressions'] = self.behaviors['impressions'].fillna('')

        if self.mode == 'train':
            self.behaviors['impressions'] = self.behaviors['impressions'].apply(lambda x: [imp.split('-') for imp in x.split(' ')])

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        behavior = self.behaviors.iloc[idx]
        clicked_news_ids = behavior['clicked_news'].split(' ')
        clicked_news = self.get_news_features(clicked_news_ids)
        
        if self.mode == 'train':
            impressions = behavior['impressions']
            labels = torch.tensor([int(imp[1]) for imp in impressions], dtype=torch.float)
            candidate_news_ids = [imp[0] for imp in impressions]
            candidate_news = self.get_news_features(candidate_news_ids)
            return clicked_news, candidate_news, labels
        else:
            impressions = behavior['impressions'].split(' ')
            candidate_news = self.get_news_features(impressions)
            return clicked_news, candidate_news

    def get_news_features(self, news_ids, max_title_length=30, max_body_length=100):
        news_df = self.news[self.news['news_id'].isin(news_ids)]
        titles = news_df['title'].fillna('').apply(lambda x: self.tokenize_title(x, max_title_length)).tolist()
        bodies = news_df['abstract'].fillna('').apply(lambda x: self.tokenize_body(x, max_body_length)).tolist()
        categories = news_df['category'].fillna('').apply(self.tokenize_category).tolist()
        return titles, bodies, categories

    def tokenize_title(self, title, max_length):
        tokens = [self.word_dict.get(word, 0) for word in title.split(' ')]
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))  # Pad
        else:
            tokens = tokens[:max_length]  # Truncate
        return tokens

    def tokenize_body(self, body, max_length):
        tokens = [self.word_dict.get(word, 0) for word in body.split(' ')]
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))  # Pad
        else:
            tokens = tokens[:max_length]  # Truncate
        return tokens

    def tokenize_category(self, category):
        return self.category_dict.get(category, 0)
