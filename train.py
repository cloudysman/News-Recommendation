import torch
from torch.utils.data import DataLoader
from torch import nn
from models.recommendation_model import NewsRecommendationModel
from utils import MINDDataset
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def create_word_dict(news_file):
    news_df = pd.read_csv(news_file, sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
    all_words = ' '.join(news_df['title'].fillna('') + ' ' + news_df['abstract'].fillna('')).split(' ')
    word_count = Counter(all_words)
    word_dict = {word: i for i, (word, _) in enumerate(word_count.items(), start=1)}
    return word_dict

def create_category_dict(news_file):
    news_df = pd.read_csv(news_file, sep='\t', header=None, names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
    categories = news_df['category'].unique()
    category_dict = {category: i for i, category in enumerate(categories, start=1)}
    return category_dict

def pad_sequence(sequence, max_length, padding_value=0):
    if len(sequence) < max_length:
        sequence = sequence + [padding_value] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

def convert_to_tensor_list(data_2d_list, dtype=torch.long):
    tensor_list = []
    for sub_list in data_2d_list:
        if isinstance(sub_list, list):
            tensor_list.append(torch.tensor(sub_list, dtype=dtype))
        else:
            print("Invalid sub_list:", sub_list)
    return tensor_list

def collate_fn(batch):
    clicked_news, candidate_news, labels = zip(*batch)

    max_title_length = 30
    max_body_length = 100

    clicked_news_padded = [[pad_sequence(news, max_length=max_body_length) for news in user_news] for user_news in clicked_news]
    candidate_news_padded = [[pad_sequence(news, max_length=max_body_length) for news in user_news] for user_news in candidate_news]

    try:
        clicked_news_tensors = convert_to_tensor_list([news for user_news in clicked_news_padded for news in user_news])
        candidate_news_tensors = convert_to_tensor_list([news for user_news in candidate_news_padded for news in user_news])
    except Exception as e:
        print("Error while converting to tensor:", e)
        print("clicked_news_padded:", clicked_news_padded)
        print("candidate_news_padded:", candidate_news_padded)
        raise e

    try:
        clicked_news_stacked = torch.stack(clicked_news_tensors).view(len(batch), -1, max_body_length)
        candidate_news_stacked = torch.stack(candidate_news_tensors).view(len(batch), -1, max_body_length)
    except Exception as e:
        print("Error while stacking tensors:", e)
        print("clicked_news_tensors:", clicked_news_tensors)
        print("candidate_news_tensors:", candidate_news_tensors)
        raise e

    labels_stacked = torch.tensor(labels, dtype=torch.float)

    return clicked_news_stacked, candidate_news_stacked, labels_stacked

def compute_similarity(news_embedding, candidate_embeddings):
    news_embedding_np = news_embedding.detach().cpu().numpy()
    candidate_embeddings_np = [embedding.detach().cpu().numpy() for embedding in candidate_embeddings]
    similarities = cosine_similarity([news_embedding_np], candidate_embeddings_np)
    return similarities

word_emb_dim = 300
num_filters = 384
window_size = 3
category_emb_dim = 100
num_embeddings = 50000
num_categories = 50
num_epochs = 10
batch_size = 100


word_dict = create_word_dict('data/MINDsmall_train/news.tsv')
category_dict = create_category_dict('data/MINDsmall_train/news.tsv')


train_dataset = MINDDataset('data/MINDsmall_train/behaviors.tsv', 'data/MINDsmall_train/news.tsv', word_dict, category_dict, mode='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


model = NewsRecommendationModel(word_emb_dim, num_filters, window_size, category_emb_dim, num_embeddings, num_categories)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()


for epoch in range(num_epochs):
    for batch in train_loader:
        clicked_news, candidate_news, labels = batch
        title, body, category = candidate_news
        clicked_title, clicked_body, clicked_category = clicked_news

        optimizer.zero_grad()
        click_prob = model(title, body, category, clicked_news)
        loss = loss_fn(click_prob, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


for batch in train_loader:
    clicked_news, candidate_news, labels = batch
    title, body, category = candidate_news
    clicked_title, clicked_body, clicked_category = clicked_news

    with torch.no_grad():
        candidate_news_embeddings = [model.get_news_embedding(t, b, c, None) for t, b, c in zip(title, body, category)]
        clicked_news_embeddings = [model.get_news_embedding(ct, cb, cc, candidate_news_embeddings) for ct, cb, cc in zip(clicked_title, clicked_body, clicked_category)]

    for news_embedding in clicked_news_embeddings:
        similarities = compute_similarity(news_embedding, candidate_news_embeddings)
        print("Similarities:", similarities)
