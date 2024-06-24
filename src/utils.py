from datetime import datetime
from typing import List

import numpy as np
import polars as pl

import torch
from torch import nn

MIN_DT = datetime(2021, 1, 1)


def prepare_data(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(
        mon_ind=12 * (pl.col("event_time").dt.year() - MIN_DT.year) + (pl.col("event_time").dt.month() - MIN_DT.month),
        day_ind=(pl.col("event_time") - MIN_DT).dt.days(),
    )
    data = data.with_columns((pl.col("event_time") - MIN_DT).dt.seconds())
    return data


def to_records(data: pl.DataFrame, embedding_col: str = "embedding", embedding_dim: int = None) -> List[dict]:
    res = [{} for _ in range(len(data))]
    for i, value in enumerate(data["target"]):
        res[i]["target"] = torch.tensor(value.to_numpy())
    if embedding_col in data.columns and embedding_dim is not None:
        for i, value in enumerate(data[embedding_col]):
            if value is None:
                res[i][embedding_col] = torch.zeros((1, embedding_dim), dtype=torch.float32)
            else:
                res[i][embedding_col] = torch.tensor(np.vstack(value.to_numpy()))
    for col, dtype in zip(data.columns, data.dtypes):
        if col in ("client_id", "target", embedding_col):
            continue
        assert dtype == pl.List
        for i, value in enumerate(data[col].fill_null([])):
            res[i][col] = torch.tensor(value.to_numpy())
    return res


def init_weights(module: nn.Module, initializer_range: float) -> None:
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.uniform_(-initializer_range, initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
