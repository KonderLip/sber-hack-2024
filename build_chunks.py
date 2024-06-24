import os
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from src import config

train_target = pd.read_parquet("./data/train_target.parquet/")
train_clients = train_target["client_id"].unique()
np.random.seed(56)
np.random.shuffle(train_clients)


def build_chunks(clients, source_dir, out_dir, n_chunks):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    chunk_size = (len(clients) + n_chunks - 1) // n_chunks
    for i in tqdm(range(n_chunks)):
        cur_clients = clients[i * chunk_size:(i + 1) * chunk_size]
        chunk = []
        for file in os.listdir(source_dir):
            data = pd.read_parquet(os.path.join(source_dir, file))
            chunk.append(data[data.client_id.isin(cur_clients)])
        chunk = pd.concat(chunk)
        chunk.to_parquet(os.path.join(out_dir, f"part-{i}.parquet"), index=False)


def main():
    build_chunks(train_clients, "./data/trx_train.parquet/", "./data/chunks/trx_train.parquet/", config.N_CHUNKS)
    build_chunks(train_clients, "./data/geo_train.parquet/", "./data/chunks/geo_train.parquet/", config.N_CHUNKS)
    build_chunks(train_clients, "./data/dial_train.parquet/", "./data/chunks/dial_train.parquet/", config.N_CHUNKS)


if __name__ == '__main__':
    main()
