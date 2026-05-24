import argparse
import itertools
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate


seed = 2020
random.seed(seed)

parser = argparse.ArgumentParser(description='swing recall')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test_swing.log')
parser.add_argument('--test_size', type=int, default=1000)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--max_users_per_item', type=int, default=800)
parser.add_argument('--sim_topk', type=int, default=300)
parser.add_argument('--recall_num', type=int, default=100)
parser.add_argument('--recent_num', type=int, default=3)
parser.add_argument('--recent_decay', type=float, default=0.7)

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'swing recall, mode: {mode}')


def get_user_item_dict(df):
    user_item = df.groupby('user_id')['click_article_id'].agg(list).reset_index()
    return dict(zip(user_item['user_id'], user_item['click_article_id']))


def get_item_user_dict(user_item_dict):
    item_user_dict = defaultdict(list)
    for user_id, items in user_item_dict.items():
        for item in set(items):
            item_user_dict[item].append(user_id)
    return item_user_dict


def cal_swing_sim(user_item_dict, item_user_dict, alpha=1.0, max_users_per_item=800):
    user_item_set = {user: set(items) for user, items in user_item_dict.items()}
    item_cnt = {item: len(users) for item, users in item_user_dict.items()}
    sim_dict = {}

    for item, users in tqdm(item_user_dict.items()):
        sim_dict.setdefault(item, defaultdict(float))

        # Very hot items create too many user pairs. Keep shorter-history users
        # first because their co-click overlap is usually cleaner.
        if len(users) > max_users_per_item:
            users = sorted(users, key=lambda u: len(user_item_set[u]))[:max_users_per_item]

        for user1, user2 in itertools.combinations(users, 2):
            common_items = user_item_set[user1] & user_item_set[user2]
            if len(common_items) <= 1:
                continue

            swing_weight = 1.0 / (alpha + len(common_items))
            for relate_item in common_items:
                if relate_item == item:
                    continue
                sim_dict[item][relate_item] += swing_weight

        # A light item-degree normalization keeps the score scale comparable
        # across hot and long-tail anchor items.
        for relate_item, score in list(sim_dict[item].items()):
            sim_dict[item][relate_item] = score / np.sqrt(
                item_cnt[item] * item_cnt.get(relate_item, 1)
            )

    return {item: dict(relate_items) for item, relate_items in sim_dict.items()}


def recall(df_query, swing_sim, user_item_dict, sim_topk=300, recall_num=100,
           recent_num=3, recent_decay=0.7):
    data_list = []

    for user_id, target_item in tqdm(df_query.values):
        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]
        interacted_set = set(interacted_items)
        recent_items = interacted_items[::-1][:recent_num]
        rank = defaultdict(float)

        for loc, item in enumerate(recent_items):
            if item not in swing_sim:
                continue

            sim_items = sorted(
                swing_sim[item].items(),
                key=lambda x: x[1],
                reverse=True
            )[:sim_topk]

            for relate_item, sim_score in sim_items:
                if relate_item in interacted_set:
                    continue
                rank[relate_item] += sim_score * (recent_decay ** loc)

        sim_items = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:recall_num]
        if not sim_items:
            continue

        df_temp = pd.DataFrame({
            'user_id': user_id,
            'article_id': [item for item, _ in sim_items],
            'sim_score': [score for _, score in sim_items],
        })

        if target_item == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == target_item, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')
        data_list.append(df_temp)

    if not data_list:
        return pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])

    return pd.concat(data_list, sort=False)


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        sim_pkl_file = '../user_data/sim/offline/swing_sim.pkl'
        output_file = '../user_data/data/offline/recall_swing.pkl'
    elif mode == 'test':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        test_users = df_query['user_id'].sample(n=args.test_size, random_state=seed)
        df_query = df_query[df_query['user_id'].isin(test_users)]
        df_click = df_click[df_click['user_id'].isin(test_users)]

        sim_pkl_file = '../user_data/sim/test/swing_sim.pkl'
        output_file = '../user_data/data/test/recall_swing.pkl'
        log.info(f'test mode users: {args.test_size}')
        log.info(f'df_click shape: {df_click.shape}')
        log.info(f'df_query shape: {df_query.shape}')
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')
        sim_pkl_file = '../user_data/sim/online/swing_sim.pkl'
        output_file = '../user_data/data/online/recall_swing.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    user_item_dict = get_user_item_dict(df_click)
    item_user_dict = get_item_user_dict(user_item_dict)

    swing_sim = cal_swing_sim(
        user_item_dict,
        item_user_dict,
        alpha=args.alpha,
        max_users_per_item=args.max_users_per_item
    )

    os.makedirs(os.path.dirname(sim_pkl_file), exist_ok=True)
    with open(sim_pkl_file, 'wb') as f:
        pickle.dump(swing_sim, f)

    df_data = recall(
        df_query,
        swing_sim,
        user_item_dict,
        sim_topk=args.sim_topk,
        recall_num=args.recall_num,
        recent_num=args.recent_num,
        recent_decay=args.recent_decay
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
        log.debug(f'swing: {metrics}')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_data.to_pickle(output_file)
    log.info(f'results saved to {output_file}')
