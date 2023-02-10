import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import gc
from typing import Dict, List, Union, Literal, Callable

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm
import warnings


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
    :param sample_frac: Float specifying how much percent of rows each bagging sample shall use.
    :param correlation_threshold: For inbuilt feature selection PSOD will filter out all columns with a correlation
                                 that is closer to 0 than the correlation_threshold. This usually speeds up training
                                 and increases accuracy of top outliers. Iy may reduce global performance slightly.
                                 Set this value to 0 to not filter any features.
    :param transform_algorithm: String choosing how numerical columns shall be transformed. Must be any of
                                ["logarithmic", "yeo-johnson", "none", None].
    :param random_seed: Int specifying the start random_seed. Each additional iteration will use a different seed.
    :param cat_encode_on_sample: If True categorical encoding will be applied to bagging sample. If False fits the
                                 encoder on the full dataset. Encoding on sample might reduce accuracy.
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
            sample_frac: float = 1.0,
            correlation_threshold: float = 0.05,
            transform_algorithm: Union[Literal["logarithmic", "yeo-johnson", "none"], None] = "logarithmic",
            random_seed: int = 1,
            cat_encode_on_sample: bool = False,
            flag_outlier_on: Literal["low end", "both ends", "high end"] = "both ends"
    ):
        self.cat_columns = cat_columns
        self.cat_encoders: Dict[Union[str, int, float], TargetEncoder] = {}
        self.numeric_encoders: Union[PowerTransformer, None] = None
        self.regressors: Dict[Union[str, int, float], LinearRegression] = {}
        self.n_jobs = n_jobs
        self.scores: Union[pd.Series, None] = None
        self.outlier_classes = Union[pd.Series, None]
        self.min_cols_chosen: Union[int, float] = min_cols_chosen
        self.max_cols_chosen: Union[int, float] = max_cols_chosen
        self.chosen_columns: Dict[Union[str, int, float]] = {}
        self.cols_with_var: List[Union[str, int, float]] = []
        self.stdevs_to_outlier = stdevs_to_outlier
        self.sample_frac = sample_frac
        self.correlation_threshold = correlation_threshold
        self.transform_algorithm = transform_algorithm
        self.flag_outlier_on = flag_outlier_on
        self.random_seed = random_seed
        self.cat_encode_on_sample = cat_encode_on_sample
        self.random_generator = np.random.default_rng(self.random_seed)
        self.pred_distribution_stats: Dict[str, float] = {}

        if self.max_cols_chosen > 1.0:
            raise ValueError("Param max_cols_chosen cannot be higher than 1.")

        if self.min_cols_chosen <= 0:
            raise ValueError("Param min_cols_chosen must be higher than 0.")

        if self.correlation_threshold >= 1.0 or self.correlation_threshold < 0:
            raise ValueError("Param correlation_threshold must be between >= 0 and < 1.")

        if self.min_cols_chosen > self.max_cols_chosen:
            raise ValueError("Param min_cols_chosen cannot be higher than param max_cols_chosen.")

        if self.flag_outlier_on not in ["low end", "both ends", "high end"]:
            raise ValueError('Param flag_outlier_on must be any of ["low end", "both ends", "high end"].')

        if self.sample_frac > 1.0:
            warning_message = """Param sample_frac has been set to higher than 1.0. This might lead to overfitting. It
             is recommended to leave this param at 1."""
            warnings.warn(warning_message, UserWarning)

        if self.min_cols_chosen < 0.3:
            warning_message = """Param min_cols_chosen has been set to a very low value of less than 0.3.
            Depending on the dataset this may reduce performance. The more columns the data has the safer it is
            to reduce this param."""
            warnings.warn(warning_message, UserWarning)

        if self.correlation_threshold > 0.15:
            warning_message = """Param correlation_threshold has been set higher than 0.15. This may harm performance or
            lead to no feature selection effectively. It is recommended to not set the value above 0.05."""
            warnings.warn(warning_message, UserWarning)

    def __str__(self):
        message = f"""
        Most important params specified are:
        - n_jobs: {self.n_jobs}
        - cat_columns: {self.cat_columns}
        - min_cols_chosen: {self.min_cols_chosen}
        - max_cols_chosen: {self.max_cols_chosen}
        - stdevs_to_outlier: {self.stdevs_to_outlier}
        - sample_frac: {self.sample_frac}
        - self.correlation_threshold: {self.correlation_threshold}
        - transform_algorithm: {self.transform_algorithm}
        - random_seed: {self.random_seed}
        - cat_encode_on_sample: {self.cat_encode_on_sample}
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

    def correlation_feature_selection(self, df, target_col):
        cols = []
        for col in df.columns:
            corr = df[col].corr(df[target_col])
            if corr > self.correlation_threshold or corr < -self.correlation_threshold and col != target_col:
                cols.append(col)
        return cols

    def col_intersection(self, lst1, lst2) -> list:
        return np.intersect1d(lst1, lst2).tolist()

    def make_outlier_classes(self, df_scores: pd.DataFrame, use_trained_stats=True):
        if not use_trained_stats:
            mean_score = df_scores["anomaly"].mean()
            std_score = df_scores["anomaly"].std()
            self.pred_distribution_stats["mean_score"] = mean_score
            self.pred_distribution_stats["std_score"] = std_score

        if self.flag_outlier_on == "both ends":
            conditions = [
                df_scores["anomaly"] < self.pred_distribution_stats["mean_score"] - self.stdevs_to_outlier * self.pred_distribution_stats["std_score"],
                df_scores["anomaly"] > self.pred_distribution_stats["mean_score"] + self.stdevs_to_outlier * self.pred_distribution_stats["std_score"]
            ]
        elif self.flag_outlier_on == "low end":
            conditions = [
                df_scores["anomaly"] < self.pred_distribution_stats["mean_score"] - self.stdevs_to_outlier * self.pred_distribution_stats["std_score"]
            ]
        elif self.flag_outlier_on == "high end":
            conditions = [
                df_scores["anomaly"] > self.pred_distribution_stats["mean_score"] + self.stdevs_to_outlier * self.pred_distribution_stats["std_score"]
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

    def fit_transform_numeric_data(self, df):
        if self.transform_algorithm == "logarithmic" and isinstance(self.cat_columns, list):
            df = np.log1p(df)
        elif self.transform_algorithm == "yeo-johnson" and isinstance(self.cat_columns, list):
            scaler = PowerTransformer(method="yeo-johnson")
            df = pd.DataFrame(
                scaler.fit_transform(df),
                columns=df.drop(self.cat_columns, axis=1).columns
            )
            self.numeric_encoders = scaler
        else:
            return df
        return df

    def transform_numeric_data(self, df):
        if self.transform_algorithm == "logarithmic" and isinstance(self.cat_columns, list):
            df = np.log1p(df)
        elif self.transform_algorithm == "yeo-johnson" and isinstance(self.cat_columns, list):
            scaler = self.numeric_encoders
            df = pd.DataFrame(
                scaler.transform(df),
                columns=df.drop(self.cat_columns, axis=1).columns
            )
        else:
            return df
        return df

    def remove_zero_variance(self, df) -> list:
        var_data: pd.Series = df.var()
        return var_data.loc[var_data != 0].index.to_list()

    def fit_predict(self, df, return_class=False) -> pd.Series:
        """
        Train PSOD and return outlier predictions.

        :param df: Pandas DataFrame to detect outliers from.
        :param return_class: Boolean indicating if class or outlier scores shall be returned. Default is False.
        :return: Returns a Pandas Series
        """
        if isinstance(self.cat_columns, list):
            self.cols_with_var = self.remove_zero_variance(df.drop(self.cat_columns, axis=1))
            df = df.loc[:, self.cols_with_var+self.cat_columns]
        else:
            self.cols_with_var = self.remove_zero_variance(df)
            df = df.loc[:, self.cols_with_var]

        df_scores = df.copy()
        self.get_range_cols(df)
        if isinstance(self.cat_columns, list):
            loop_cols = df.drop(self.cat_columns, axis=1).columns
            df.drop(self.cat_columns, axis=1).loc[:, :] = self.fit_transform_numeric_data(
                df.drop(self.cat_columns, axis=1).loc[:, :]
            )
        else:
            loop_cols = df.columns
            df.loc[:, :] = self.fit_transform_numeric_data(df.loc[:, :])

        for enum, col in tqdm(enumerate(loop_cols), total=len(loop_cols)):
            self.chosen_columns[col] = self.chose_random_columns(df.drop(col, axis=1))
            temp_df = df.copy()
            # encode categorical columns that are in chosen columns
            if isinstance(self.cat_columns, list):
                chosen_cat_cols = self.col_intersection(
                    self.cat_columns, self.chosen_columns[col]
                )
            else:
                chosen_cat_cols = self.chosen_columns[col]

            if isinstance(self.cat_columns, list):
                corr_cols = self.correlation_feature_selection(temp_df.drop(self.cat_columns, axis=1), col)
                corr_cols = self.col_intersection(
                    corr_cols, self.chosen_columns[col]
                )
            else:
                corr_cols = self.correlation_feature_selection(temp_df, col)
                corr_cols = self.col_intersection(
                    corr_cols, self.chosen_columns[col]
                )

            if len(corr_cols) > 0:
                self.chosen_columns[col] = corr_cols

            idx = df_scores.sample(frac=self.sample_frac, random_state=enum, replace=True).index

            if isinstance(self.cat_columns, list):
                enc = TargetEncoder(cols=chosen_cat_cols)
                if self.cat_encode_on_sample:
                    enc.fit(
                        df.loc[:, chosen_cat_cols].iloc[idx].reset_index(drop=True),
                        df.loc[:, col].iloc[idx].reset_index(drop=True),
                    )
                else:
                    enc.fit(
                        df.loc[:, chosen_cat_cols].reset_index(drop=True),
                        df.loc[:, col].reset_index(drop=True),
                    )

                temp_df.loc[:, chosen_cat_cols] = enc.transform(
                    df.loc[:, chosen_cat_cols],
                    df.loc[:, col],
                )

            reg = LinearRegression(n_jobs=self.n_jobs).fit(
                temp_df.loc[:, self.chosen_columns[col]].iloc[idx].reset_index(drop=True),
                temp_df[col].iloc[idx].reset_index(drop=True),
            )
            df_scores[col] = reg.predict(temp_df.loc[:, self.chosen_columns[col]])
            df_scores[col] = abs(temp_df[col].values - df_scores[col].values)

            self.regressors[col] = reg
            if isinstance(self.cat_columns, list):
                self.cat_encoders[col] = enc
                del enc

            del temp_df
            del idx
            del reg
            _ = gc.collect()

        df_scores = self.drop_cat_columns(df_scores)

        if return_class:
            return self.make_outlier_classes(df_scores, use_trained_stats=False)
        else:
            return df_scores["anomaly"]

    def predict(self, df, return_class=False, use_trained_stats=True) -> pd.Series:
        """
        Use trained PSOD instance to predict outliers on new data.

        :param df: Pandas DataFrame to predict outliers from.
        :param return_class: Boolean indicating if class or outlier scores shall be returned. Default is False.
        :param use_trained_stats: Boolean indicating of conversion from outlier scores to outlier class shall make use
        of mean and std of prediction errors obtained during training shall be used. If False prediction errors
        of the provided dataset will be treated as new distribution with new mean and std as classification thresholds.
        :return: Returns a Pandas Series
        """
        if isinstance(self.cat_columns, list):
            df = df.loc[:, self.cols_with_var+self.cat_columns]
        else:
            df = df.loc[:, self.cols_with_var]
        df_scores = df.copy()

        if isinstance(self.cat_columns, list):
            loop_cols = df.drop(self.cat_columns, axis=1).columns
            df.drop(self.cat_columns, axis=1).loc[:, :] = self.transform_numeric_data(
                df.drop(self.cat_columns, axis=1).loc[:, :]
            )
        else:
            loop_cols = df.columns
            df = self.transform_numeric_data(df.loc[:, :])

        for enum, col in tqdm(enumerate(loop_cols)):
            temp_df = df
            if isinstance(self.cat_columns, list):
                chosen_cat_cols = self.col_intersection(
                    self.cat_columns, self.chosen_columns[col]
                )
            else:
                chosen_cat_cols = self.chosen_columns[col]

            if isinstance(self.cat_columns, list):
                enc = self.cat_encoders[col]
                temp_df[chosen_cat_cols] = enc.transform(df[chosen_cat_cols])

            reg = self.regressors[col]

            df_scores[col] = reg.predict(df[self.chosen_columns[col]])
            df_scores[col] = abs(df[col] - df_scores[col])
            self.regressors[col] = reg

        df_scores = self.drop_cat_columns(df_scores)

        if return_class:
            return self.make_outlier_classes(df_scores, use_trained_stats=use_trained_stats)
        else:
            return df_scores["anomaly"]
