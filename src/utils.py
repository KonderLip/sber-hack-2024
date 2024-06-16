from datetime import datetime
from tqdm.auto import tqdm
from typing import List

import polars as pl
import torch

min_dt = datetime(2021, 1, 1)


def prepare_data(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(
        mon_ind=12 * (pl.col("event_time").dt.year() - min_dt.year) + (pl.col("event_time").dt.month() - min_dt.month),
        day_ind=(pl.col("event_time") - min_dt).dt.days(),
    )
    data = data.with_columns(
        (pl.col("event_time") - min_dt).dt.seconds()
    )
    return data


def to_records(data: pl.DataFrame) -> List[dict]:
    res = [{} for _ in range(len(data))]
    for i, value in enumerate(data["target"]):
        res[i]["target"] = value.to_list()
    for col, dtype in zip(tqdm(data.columns), data.dtypes):
        if col in ("client_id", "target"):
            continue
        assert dtype == pl.List
        for i, value in enumerate(data[col].fill_null([])):
            res[i][col] = torch.tensor(value.to_numpy())
    return res
