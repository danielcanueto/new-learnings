#!/usr/bin/env python
# coding: utf-8
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal


def add_features(df):
    df['question_text'] = df['question_text'].apply(lambda x: str(x))
    df["lower_question_text"] = df["question_text"].apply(
        lambda x: x.lower())
    df['total_length'] = df['question_text'].apply(len)
    df['capitals'] = df['question_text'].apply(
        lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(
        lambda row: float(row['capitals']) / float(row['total_length']),
        axis=1)
    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].apply(
        lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['num_exclamation_marks'] = df['question_text'].apply(
        lambda comment: comment.count('!'))
    df['num_question_marks'] = df['question_text'].apply(
        lambda comment: comment.count('?'))
    df['num_punctuation'] = df['question_text'].apply(
        lambda comment: sum(comment.count(w) for w in '.,;:'))
    df['num_symbols'] = df['question_text'].apply(
        lambda comment: sum(comment.count(w) for w in '*&$%'))
    df['num_smilies'] = df['question_text'].apply(lambda comment: sum(
        comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    df['num_sad'] = df['question_text'].apply(lambda comment: sum(
        comment.count(w) for w in (':-<', ':()', ';-()', ';(')))
    df["mean_word_len"] = df["question_text"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]))
    return df


def parallelize_dataframe(df, func, n_cores=9):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def async_parallelize_dataframe(df, func, n_cores=9):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pool.map_async(func, df_split)
    pool.close()
    pool.join()
    return df


# In[3]:
if __name__ == '__main__':

    #Quora insincere classification train.csv
    train_df = pd.read_csv('train.csv')

    start_time = time.time()

    preproc_df_v1 = add_features(train_df.sample(1000000, random_state=1))

    print(time.time() - start_time)
    start_time = time.time()

    preproc_df_v2 = parallelize_dataframe(
        train_df.sample(1000000, random_state=1), add_features,
        n_cores=cpu_count())
    print(time.time() - start_time)

    assert_frame_equal(preproc_df_v1, preproc_df_v2)

    # start_time = time.time()

    # preproc_df_v3 = async_parallelize_dataframe(
    #     train_df.sample(10000, random_state=1), add_features,
    #     n_cores=cpu_count())
    # print(time.time() - start_time)
    #
    # print(preproc_df_v3)
    #
    # assert_frame_equal(preproc_df_v1, preproc_df_v3)
