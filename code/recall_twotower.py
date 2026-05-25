import argparse
import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        'recall_twotower.py requires PyTorch. Please install torch before '
        'running the two-tower recall.'
    ) from exc


warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='two-tower recall')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test_twotower.log')
parser.add_argument('--test_size', type=int, default=1000)
parser.add_argument('--max_seq_len', type=int, default=20)
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--cate_dim', type=int, default=16)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--content_emb_dim', type=int, default=250)
parser.add_argument('--content_emb_path', default='../data/articles_emb.csv')
parser.add_argument('--content_emb_chunksize', type=int, default=50000)
parser.add_argument('--user_time_decay', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--max_samples', type=int, default=800000)
parser.add_argument('--recall_num', type=int, default=100)
parser.add_argument('--recall_batch_size', type=int, default=512)
parser.add_argument('--annoy_trees', type=int, default=50, help='kept for CLI compatibility; exact batched retrieval is used')
parser.add_argument('--index_all_articles', action='store_true',
                    help='index every article in articles.csv instead of active clicked articles')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
device = torch.device(args.device)

os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'two-tower recall, mode: {mode}, device: {device}')


class SequenceDataset(Dataset):
    def __init__(self, histories, targets):
        self.histories = torch.from_numpy(histories).long()
        self.targets = torch.from_numpy(targets).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.histories[idx], self.targets[idx]


class TwoTowerModel(nn.Module):
    def __init__(self, item_num, cate_num, emb_dim=64, cate_dim=16,
                 hidden_dim=128, content_emb_dim=250, max_seq_len=20,
                 user_time_decay=0.9):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.user_time_decay = user_time_decay
        self.item_embedding = nn.Embedding(item_num + 1, emb_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(cate_num + 1, cate_dim, padding_idx=0)

        self.user_attention = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.content_projection = nn.Sequential(
            nn.Linear(content_emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim + cate_dim + emb_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def encode_user(self, hist_items):
        hist_emb = self.item_embedding(hist_items)
        mask = (hist_items > 0)
        attn_logits = self.user_attention(hist_emb).squeeze(-1)

        pos = torch.arange(self.max_seq_len, device=hist_items.device).float()
        distance_from_recent = (self.max_seq_len - 1) - pos
        recency_bias = torch.log(
            torch.pow(
                torch.full_like(distance_from_recent, self.user_time_decay),
                distance_from_recent
            ).clamp(min=1e-6)
        )
        attn_logits = attn_logits + recency_bias.unsqueeze(0)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weight = F.softmax(attn_logits, dim=1).unsqueeze(-1)
        pooled = (hist_emb * attn_weight).sum(dim=1)
        user_vec = self.user_mlp(pooled)
        return F.normalize(user_vec, p=2, dim=1)

    def encode_item(self, item_ids, item_cates, item_dense, item_content):
        item_emb = self.item_embedding(item_ids)
        cate_emb = self.category_embedding(item_cates)
        content_emb = self.content_projection(item_content)
        item_input = torch.cat([item_emb, cate_emb, content_emb, item_dense], dim=1)
        item_vec = self.item_mlp(item_input)
        return F.normalize(item_vec, p=2, dim=1)

    def forward(self, hist_items, target_items, item_cates, item_dense,
                item_content, temperature):
        user_vec = self.encode_user(hist_items)
        item_vec = self.encode_item(target_items, item_cates, item_dense, item_content)
        logits = torch.matmul(user_vec, item_vec.t()) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss


def get_paths(run_mode):
    if run_mode == 'valid':
        return {
            'click': '../user_data/data/offline/click.pkl',
            'query': '../user_data/data/offline/query.pkl',
            'data_dir': '../user_data/data/offline',
            'model_dir': '../user_data/model/offline',
        }
    if run_mode == 'test':
        return {
            'click': '../user_data/data/offline/click.pkl',
            'query': '../user_data/data/offline/query.pkl',
            'data_dir': '../user_data/data/test',
            'model_dir': '../user_data/model/test',
        }
    return {
        'click': '../user_data/data/online/click.pkl',
        'query': '../user_data/data/online/query.pkl',
        'data_dir': '../user_data/data/online',
        'model_dir': '../user_data/model/online',
    }


def minmax(values):
    values = values.astype('float32')
    v_min = values.min()
    v_max = values.max()
    if v_max == v_min:
        return np.zeros_like(values, dtype='float32')
    return (values - v_min) / (v_max - v_min)


def build_article_features(df_article, candidate_articles):
    article_ids = sorted(set([int(x) for x in candidate_articles]))
    article_to_idx = {article_id: idx + 1 for idx, article_id in enumerate(article_ids)}
    idx_to_article = {idx: article_id for article_id, idx in article_to_idx.items()}

    df_article = df_article.copy()
    df_article['article_id'] = df_article['article_id'].astype(int)
    df_article = df_article.drop_duplicates('article_id')
    df_article = df_article.set_index('article_id').reindex(article_ids).reset_index()
    df_article['category_id'] = df_article['category_id'].fillna(-1).astype(int)
    df_article['created_at_ts'] = df_article['created_at_ts'].fillna(df_article['created_at_ts'].median())
    df_article['words_count'] = df_article['words_count'].fillna(df_article['words_count'].median())

    cate_values = sorted(df_article['category_id'].unique().tolist())
    cate_to_idx = {cate: idx + 1 for idx, cate in enumerate(cate_values)}

    item_cates = np.zeros(len(article_ids) + 1, dtype='int64')
    item_dense = np.zeros((len(article_ids) + 1, 2), dtype='float32')
    created_scaled = minmax(df_article['created_at_ts'].values)
    words_scaled = minmax(df_article['words_count'].values)

    for row_idx, row in df_article.iterrows():
        article_id = int(row['article_id'])
        item_idx = article_to_idx[article_id]
        item_cates[item_idx] = cate_to_idx[int(row['category_id'])]
        item_dense[item_idx, 0] = created_scaled[row_idx]
        item_dense[item_idx, 1] = words_scaled[row_idx]

    return article_to_idx, idx_to_article, item_cates, item_dense, len(cate_to_idx)


def load_article_content_embeddings(article_to_idx):
    item_content = np.zeros(
        (len(article_to_idx) + 1, args.content_emb_dim),
        dtype='float32'
    )
    if args.content_emb_dim <= 0:
        return item_content

    if not os.path.exists(args.content_emb_path):
        log.warning(f'content embedding file not found: {args.content_emb_path}')
        return item_content

    candidate_articles = set(article_to_idx.keys())
    header = pd.read_csv(args.content_emb_path, nrows=0)
    emb_cols = [col for col in header.columns if col.startswith('emb_')]
    emb_cols = sorted(emb_cols, key=lambda x: int(x.split('_')[1]))
    emb_cols = emb_cols[:args.content_emb_dim]
    if len(emb_cols) != args.content_emb_dim:
        log.warning(
            f'expected {args.content_emb_dim} content embedding columns, got {len(emb_cols)}'
        )

    loaded_cnt = 0
    usecols = ['article_id'] + emb_cols
    dtype = {col: 'float32' for col in emb_cols}
    for chunk in tqdm(
        pd.read_csv(
            args.content_emb_path,
            usecols=usecols,
            dtype=dtype,
            chunksize=args.content_emb_chunksize
        ),
        desc='loading article content embeddings'
    ):
        chunk['article_id'] = chunk['article_id'].astype(int)
        chunk = chunk[chunk['article_id'].isin(candidate_articles)]
        if chunk.empty:
            continue

        vectors = chunk[emb_cols].values.astype('float32')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        for article_id, vector in zip(chunk['article_id'].values, vectors):
            item_content[article_to_idx[int(article_id)]] = vector
            loaded_cnt += 1

    log.info(
        f'loaded content embeddings for {loaded_cnt}/{len(article_to_idx)} candidate articles'
    )
    return item_content


def build_user_item_dict(df_click):
    df_click = df_click.sort_values(['user_id', 'click_timestamp'])
    user_item = df_click.groupby('user_id')['click_article_id'].agg(list).reset_index()
    return dict(zip(user_item['user_id'], user_item['click_article_id']))


def build_training_arrays(user_item_dict, article_to_idx, max_seq_len, max_samples):
    samples = []
    for _, article_seq in tqdm(user_item_dict.items(), desc='building train samples'):
        seq = [article_to_idx[item] for item in article_seq if item in article_to_idx]
        if len(seq) < 2:
            continue
        for pos in range(1, len(seq)):
            hist = seq[max(0, pos - max_seq_len):pos]
            target = seq[pos]
            samples.append((hist, target))

    if max_samples > 0 and len(samples) > max_samples:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_samples)

    histories = np.zeros((len(samples), max_seq_len), dtype='int64')
    targets = np.zeros(len(samples), dtype='int64')
    for i, (hist, target) in enumerate(samples):
        histories[i, -len(hist):] = hist
        targets[i] = target

    return histories, targets


def train_model(model, histories, targets, item_cates, item_dense, item_content):
    dataset = SequenceDataset(histories, targets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    item_cates_tensor = torch.from_numpy(item_cates).long().to(device)
    item_dense_tensor = torch.from_numpy(item_dense).float().to(device)
    item_content_tensor = torch.from_numpy(item_content).float().to(device)

    model.to(device)
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        step = 0
        for hist_items, target_items in tqdm(loader, desc=f'train epoch {epoch + 1}'):
            hist_items = hist_items.to(device)
            target_items = target_items.to(device)
            loss = model(
                hist_items,
                target_items,
                item_cates_tensor[target_items],
                item_dense_tensor[target_items],
                item_content_tensor[target_items],
                args.temperature
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

        log.info(f'epoch {epoch + 1}, loss: {total_loss / max(step, 1):.6f}')

    return model


def encode_all_items(model, item_cates, item_dense, item_content, idx_to_article, batch_size=4096):
    model.eval()
    item_indices = np.array(sorted(idx_to_article.keys()), dtype='int64')
    item_vectors = {}
    item_cates_tensor = torch.from_numpy(item_cates).long().to(device)
    item_dense_tensor = torch.from_numpy(item_dense).float().to(device)
    item_content_tensor = torch.from_numpy(item_content).float().to(device)

    with torch.no_grad():
        for start in tqdm(range(0, len(item_indices), batch_size), desc='encoding items'):
            batch_idx = item_indices[start:start + batch_size]
            batch_items = torch.from_numpy(batch_idx).long().to(device)
            vec = model.encode_item(
                batch_items,
                item_cates_tensor[batch_items],
                item_dense_tensor[batch_items],
                item_content_tensor[batch_items]
            ).cpu().numpy()

            for item_idx, item_vec in zip(batch_idx, vec):
                article_id = idx_to_article[int(item_idx)]
                item_vectors[article_id] = item_vec.astype('float32')

    return item_vectors


def make_history_array(article_seq, article_to_idx, max_seq_len):
    hist = [article_to_idx[item] for item in article_seq if item in article_to_idx]
    hist = hist[-max_seq_len:]
    arr = np.zeros((1, max_seq_len), dtype='int64')
    if hist:
        arr[0, -len(hist):] = hist
    return arr


def recall(df_query, user_item_dict, article_to_idx, model, item_vectors, hot_articles):
    data_list = []
    score_cache = {}
    model.eval()

    article_ids = np.array(list(item_vectors.keys()), dtype='int64')
    item_matrix = np.stack([item_vectors[int(article_id)] for article_id in article_ids]).astype('float32')
    item_matrix = torch.from_numpy(item_matrix).float().to(device)
    article_pos = {int(article_id): pos for pos, article_id in enumerate(article_ids)}
    query_values = df_query.values

    with torch.no_grad():
        for start in tqdm(range(0, len(query_values), args.recall_batch_size), desc='recalling'):
            batch_rows = query_values[start:start + args.recall_batch_size]
            hist_arrays = []
            batch_meta = []

            for user_id, target_item in batch_rows:
                history = user_item_dict.get(user_id, [])
                hist_arr = make_history_array(history, article_to_idx, args.max_seq_len)
                if hist_arr.sum() == 0:
                    candidates = [(item, 1.0 / (rank + 1)) for rank, item in enumerate(hot_articles[:args.recall_num])]
                    append_recall_result(data_list, score_cache, user_id, target_item, candidates)
                    continue

                hist_arrays.append(hist_arr[0])
                batch_meta.append((user_id, target_item, set(history)))

            if not hist_arrays:
                continue

            hist_tensor = torch.from_numpy(np.stack(hist_arrays)).long().to(device)
            user_vec = model.encode_user(hist_tensor)
            scores = torch.matmul(user_vec, item_matrix.t())

            for row_idx, (user_id, target_item, interacted_set) in enumerate(batch_meta):
                seen_pos = [article_pos[item] for item in interacted_set if item in article_pos]
                if seen_pos:
                    scores[row_idx, torch.tensor(seen_pos, device=device)] = -1e9

            topk = min(args.recall_num, scores.size(1))
            top_scores, top_pos = torch.topk(scores, k=topk, dim=1)
            top_scores = top_scores.cpu().numpy()
            top_pos = top_pos.cpu().numpy()

            for row_idx, (user_id, target_item, interacted_set) in enumerate(batch_meta):
                candidates = []
                for pos, score in zip(top_pos[row_idx], top_scores[row_idx]):
                    article_id = int(article_ids[pos])
                    if article_id in interacted_set:
                        continue
                    candidates.append((article_id, float(score)))

                if len(candidates) < args.recall_num:
                    used = {item for item, _ in candidates} | interacted_set
                    for rank, article_id in enumerate(hot_articles):
                        if article_id in used:
                            continue
                        candidates.append((article_id, 1e-6 / (rank + 1)))
                        if len(candidates) >= args.recall_num:
                            break

                append_recall_result(data_list, score_cache, user_id, target_item, candidates)

    if not data_list:
        empty = pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])
        return empty, score_cache
    return pd.concat(data_list, sort=False), score_cache


def append_recall_result(data_list, score_cache, user_id, target_item, candidates):
    if not candidates:
        return

    df_temp = pd.DataFrame({
        'user_id': int(user_id),
        'article_id': [int(item) for item, _ in candidates],
        'sim_score': [float(score) for _, score in candidates],
    })
    if target_item == -1:
        df_temp['label'] = np.nan
    else:
        df_temp['label'] = 0
        df_temp.loc[df_temp['article_id'] == int(target_item), 'label'] = 1

    data_list.append(df_temp[['user_id', 'article_id', 'sim_score', 'label']])
    score_cache[int(user_id)] = {int(item): float(score) for item, score in candidates}


if __name__ == '__main__':
    paths = get_paths(mode)
    df_click = pd.read_pickle(paths['click'])
    df_query = pd.read_pickle(paths['query'])

    if mode == 'test':
        test_users = df_query['user_id'].sample(n=args.test_size, random_state=seed)
        df_query = df_query[df_query['user_id'].isin(test_users)]
        df_click = df_click[df_click['user_id'].isin(test_users)]

    os.makedirs(paths['data_dir'], exist_ok=True)
    os.makedirs(paths['model_dir'], exist_ok=True)

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'df_query shape: {df_query.shape}')

    df_article = pd.read_csv('../data/articles.csv')
    clicked_articles = df_click['click_article_id'].astype(int).unique().tolist()
    positive_query_articles = (
        df_query[df_query['click_article_id'] != -1]['click_article_id']
        .astype(int)
        .unique()
        .tolist()
    )
    if args.index_all_articles:
        candidate_articles = df_article['article_id'].astype(int).unique().tolist()
    else:
        candidate_articles = clicked_articles + positive_query_articles
    article_to_idx, idx_to_article, item_cates, item_dense, cate_num = build_article_features(
        df_article,
        candidate_articles
    )
    item_content = load_article_content_embeddings(article_to_idx)
    user_item_dict = build_user_item_dict(df_click)
    hot_articles = (
        df_click['click_article_id'].value_counts().head(args.recall_num * 5).index.astype(int).tolist()
    )

    histories, targets = build_training_arrays(
        user_item_dict,
        article_to_idx,
        args.max_seq_len,
        args.max_samples
    )
    log.info(f'train samples: {len(targets)}, articles: {len(article_to_idx)}, categories: {cate_num}')

    model = TwoTowerModel(
        item_num=len(article_to_idx),
        cate_num=cate_num,
        emb_dim=args.emb_dim,
        cate_dim=args.cate_dim,
        hidden_dim=args.hidden_dim,
        content_emb_dim=args.content_emb_dim,
        max_seq_len=args.max_seq_len,
        user_time_decay=args.user_time_decay
    )
    model = train_model(model, histories, targets, item_cates, item_dense, item_content)

    model_path = os.path.join(paths['model_dir'], 'twotower.pt')
    torch.save(model.state_dict(), model_path)
    log.info(f'model saved to {model_path}')

    item_vectors = encode_all_items(model, item_cates, item_dense, item_content, idx_to_article)
    with open(os.path.join(paths['data_dir'], 'article_twotower.pkl'), 'wb') as f:
        pickle.dump(item_vectors, f)

    df_data, score_cache = recall(
        df_query,
        user_item_dict,
        article_to_idx,
        model,
        item_vectors,
        hot_articles
    )

    df_data = df_data.sort_values(
        ['user_id', 'sim_score'],
        ascending=[True, False]
    ).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    if mode == 'valid':
        log.info('calculating recall metrics')
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        metrics = evaluate(df_data[df_data['label'].notnull()], total)
        log.debug(f'twotower: {metrics}')

    output_file = os.path.join(paths['data_dir'], 'recall_twotower.pkl')
    score_file = os.path.join(paths['data_dir'], 'twotower_score.pkl')
    df_data.to_pickle(output_file)
    with open(score_file, 'wb') as f:
        pickle.dump(score_cache, f)
    log.info(f'results saved to {output_file}')
