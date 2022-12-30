import gc
from typing import Dict, List, Union, Literal

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


class PSOD:
    """
    Get outlier predictions using a pseudo-supervised approach.

    :param n_jobs: Used to determine number of cores used for LinearRegression. Check sklearn documentation for details.
    :param cat_columns: None if no categorical features are present. Otherwise list specifying column names of
                          categorical features.
    :param min_cols_chosen: Float specifying the minimum percentage of columns to be used for each regressor.
    :param max_cols_chosen: Float specifying the maximum percentage of columns to be used for each regressor.
    :param stdevs_to_outlier: Float specifying after how many standard deviations the mean prediction error will be
                              flagged as an outlier.
    :param log_transform: Boolean to set if the numerical data will be log-transformed.
    :param random_seed: Int specifying the start random_seed. Each additional iteration will use a different seed.
    :param flag_outlier_on: String indicating if outliers shall we errors that are on the top end, bottom end or
                            both ends of the mean error distribution. Must be any of ["low end", "both ends", "high end"]
    """
    def __init__(
            self,
            n_jobs=-1,
            cat_columns: Union[List[str], List[int], List[float], None] = None,
            min_cols_chosen: float = 0.5,
            max_cols_chosen: float = 1.0,
            stdevs_to_outlier: float = 1.96,
            log_transform: bool = True,
            random_seed: int = 1,
            flag_outlier_on: Literal["low end", "both ends", "high end"] = "both ends"
    ):
        self.cat_columns = cat_columns
        self.cat_encoders: Dict[Union[str, int, float], TargetEncoder] = {}
        self.regressors: Dict[Union[str, int, float], LinearRegression] = {}
        self.n_jobs = n_jobs
        self.scores: Union[pd.Series, None] = None
        self.outlier_classes = Union[pd.Series, None]
        self.min_cols_chosen: Union[int, float] = min_cols_chosen
        self.max_cols_chosen: Union[int, float] = max_cols_chosen
        self.chosen_columns: List[list] = []
        self.stdevs_to_outlier = stdevs_to_outlier
        self.log_transform = log_transform
        self.flag_outlier_on = flag_outlier_on
        self.random_seed = random_seed
        self.random_generator = np.random.default_rng(self.random_seed)

        if self.max_cols_chosen > 1.0:
            raise ValueError("Param max_cols_chosen cannot be higher than 1.")

        if self.min_cols_chosen <= 0:
            raise ValueError("Param min_cols_chosen must be higher than 0.")

        if self.min_cols_chosen > self.max_cols_chosen:
            raise ValueError("Param min_cols_chosen cannot be higher than param max_cols_chosen.")

        if self.flag_outlier_on not in ["low end", "both ends", "high end"]:
            raise ValueError('Param flag_outlier_on must be any of ["low end", "both ends", "high end"].')

    def __str__(self):
        message = f"""
        Most important params specified are:
        - n_jobs: {self.n_jobs}
        - cat_columns: {self.cat_columns}
        - min_cols_chosen: {self.min_cols_chosen}
        - max_cols_chosen: {self.max_cols_chosen}
        - stdevs_to_outlier: {self.stdevs_to_outlier}
        - log_transform: {self.log_transform}
        - random_seed: {self.random_seed}
        - flag_outlier_on: {self.flag_outlier_on}
        """
        return message

    def get_range_cols(self, df):
        len_cols = len(df.columns) - 1  # taking out the "target" column
        self.min_cols_chosen: int = max(int(len_cols * self.min_cols_chosen), 1)
        self.max_cols_chosen: int = min(int(len_cols * self.max_cols_chosen), len_cols)

    def chose_random_columns(self, df) -> list:
        """
        Select random columns.

        Randomize number of columns to chose from as well as the columns chosen.
        :return: list object with chosen column names
        """
        if self.min_cols_chosen == 1 & self.max_cols_chosen == 1:
            nb_cols: int = 1
        else:
            nb_cols: int = self.random_generator.choice(
                np.arange(self.min_cols_chosen, self.max_cols_chosen) + 1, 1, replace=False
            )
        return self.random_generator.choice(df.columns, nb_cols, replace=False).tolist()

    def col_intersection(self, lst1, lst2) -> list:
        chosen_cat_cols = [value for value in lst1 if value in lst2]
        return chosen_cat_cols

    def make_outlier_classes(self, df_scores: pd.DataFrame):
        mean_score = df_scores["anomaly"].mean()
        std_score = df_scores["anomaly"].std()

        if self.flag_outlier_on == "both ends":
            conditions = [
                df_scores["anomaly"] < mean_score - self.stdevs_to_outlier * std_score,
                df_scores["anomaly"] > mean_score + self.stdevs_to_outlier * std_score
            ]
        elif self.flag_outlier_on == "low end":
            conditions = [
                df_scores["anomaly"] < mean_score - self.stdevs_to_outlier * std_score
            ]
        elif self.flag_outlier_on == "high end":
            conditions = [
                df_scores["anomaly"] > mean_score + self.stdevs_to_outlier * std_score
            ]
        else:
            raise ValueError('Param flag_outlier_on must be any of ["low end", "both ends", "high end"].')

        choices = [1 for i in conditions]
        df_scores["anomaly_class"] = np.select(conditions, choices, default=0)
        self.outlier_classes = df_scores["anomaly_class"]
        return df_scores["anomaly_class"]

    def drop_cat_columns(self, df_scores: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.cat_columns, list):
            df_scores["anomaly"] = df_scores.drop(self.cat_columns, axis=1).mean(axis=1)
        else:
            df_scores["anomaly"] = df_scores.mean(axis=1)

        self.scores = df_scores["anomaly"]
        return df_scores

    def fit_predict(self, df, return_class=False) -> pd.Series:
        df_scores = df.copy()
        self.get_range_cols(df)
        if isinstance(self.cat_columns, list):
            loop_cols = df.drop(self.cat_columns, axis=1).columns
        else:
            loop_cols = df.columns

        if self.log_transform and isinstance(self.cat_columns, list):
            df.drop(self.cat_columns, axis=1).loc[:, :] = np.log1p(
                df.drop(self.cat_columns, axis=1).loc[:, :]
            )

        for enum, col in tqdm(enumerate(loop_cols), total=len(loop_cols)):
            self.chosen_columns.append(self.chose_random_columns(df.drop(col, axis=1)))
            temp_df = df.copy()
            # encode categorical columns that are in chosen columns
            if isinstance(self.cat_columns, list):
                chosen_cat_cols = self.col_intersection(
                    self.cat_columns, self.chosen_columns[enum]
                )

            idx = df_scores.sample(frac=1.0, random_state=enum, replace=True).index

            if isinstance(self.cat_columns, list):
                enc = TargetEncoder(cols=chosen_cat_cols)
                temp_df.loc[:, chosen_cat_cols] = enc.fit_transform(
                    df.loc[:, chosen_cat_cols].iloc[idx].reset_index(drop=True),
                    df.loc[:, col].iloc[idx].reset_index(drop=True),
                )

            reg = LinearRegression(n_jobs=self.n_jobs).fit(
                temp_df.loc[:, self.chosen_columns[enum]].iloc[idx],
                temp_df[col].iloc[idx],
            )
            df_scores[col] = reg.predict(temp_df.loc[:, self.chosen_columns[enum]])
            df_scores[col] = abs(temp_df[col] - df_scores[col])
            self.regressors[col] = reg
            if isinstance(self.cat_columns, list):
                self.cat_encoders[col] = enc
            del temp_df
            _ = gc.collect()

        df_scores = self.drop_cat_columns(df_scores)

        if return_class:
            return self.make_outlier_classes(df_scores)
        else:
            return df_scores["anomaly"]

    def predict(self, df, return_class=False) -> pd.Series:
        df_scores = df.copy()

        if self.log_transform:
            df_scores.drop(self.cat_columns, axis=1).loc[:, :] = np.log1p(
                df_scores.drop(self.cat_columns, axis=1).loc[:, :]
            )

        if isinstance(self.cat_columns, list) and isinstance(self.cat_columns, list):
            loop_cols = df.drop(self.cat_columns, axis=1).columns
        else:
            loop_cols = df.columns

        for enum, col in tqdm(enumerate(loop_cols)):
            temp_df = df
            chosen_cat_cols = self.col_intersection(
                self.cat_columns, self.chosen_columns[enum]
            )
            if isinstance(self.cat_columns, list):
                enc = self.cat_encoders[col]
                temp_df[chosen_cat_cols] = enc.transform(df[chosen_cat_cols])

            reg = self.regressors[col]

            df_scores[col] = reg.predict(df[self.chosen_columns[enum]])
            df_scores[col] = abs(df[col] - df_scores[col])
            self.regressors[col] = reg

        df_scores = self.drop_cat_columns(df_scores)

        if return_class:
            return self.make_outlier_classes(df_scores)
        else:
            return df_scores["anomaly"]