from collections import defaultdict
from typing import List

import polars as pl


class PolarsDataPreprocessor:
    def __init__(
            self,
            col_id: str,
            col_event_time: str,
            cols_category: List[str] = None,
            cols_numerical: List[str] = None,
            max_seq_len: int = 512,
            prefix: str = '',
    ) -> None:
        if cols_category is None:
            cols_category = []
        if cols_numerical is None:
            cols_numerical = []

        self._col_id = col_id
        self._col_event_time = col_event_time
        self._cols_category = cols_category
        self._cols_numerical = cols_numerical
        self._max_seq_len = max_seq_len
        self._prefix = prefix

        self._encoders = defaultdict(dict)

    def fit(self, x: pl.DataFrame) -> "PolarsDataPreprocessor":
        for col in self._cols_category:
            for value in x[col]:
                if value not in self._encoders[col]:
                    self._encoders[col][value] = len(self._encoders[col]) + 1
        return self

    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        x = (
            x
            .sort(self._col_event_time)
            .group_by(self._col_id, maintain_order=True)
            .tail(self._max_seq_len)
        )
        x = x.with_columns(
            [pl.col(col).map_dict(self._encoders[col], default=0).cast(pl.Int32) for col in self._cols_category]
        )
        x = (
            x
            .group_by(self._col_id, maintain_order=True)
            .agg(
                [pl.col(self._col_event_time)] + [pl.col(col) for col in self._cols_category + self._cols_numerical]
            )
        )
        if self._prefix != '':
            x = x.rename(
                {col: f"{self._prefix}_{col}" for col in x.columns if col != self._col_id}
            )
        return x

    def fit_transform(self, x: pl.DataFrame) -> pl.DataFrame:
        self.fit(x)
        return self.transform(x)

    def get_category_dictionary_sizes(self) -> dict:
        return {key: len(value) + 1 for key, value in self._encoders.items()}
