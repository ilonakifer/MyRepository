
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from pandas.api.types import is_numeric_dtype
import pandas as pd

class DataTransformer:
    """Docstring:
    Class for transforming data into a format that can be sent to a predictor
    1. columns with redundant categories are removed
    2. Binary columns are transformed to numeric
    3. Categorical columns are transformed using one-hot encoding
    """

    def __init__(self, **kwargs):
        self.lbb = LabelBinarizer()
        self.binary_cols = []
        self.category_cols = []
        pass

    def __remove_value_redundancy(self, df_data):
        """Docstring:
        Turn categorical columns that contain redundant info into binary columns,
        since another column already contains the third category info.

        This function is a data-specific patch that improves results a little, but it can easily be generalized:
        Whenever there is a case where two column values always co-appear together in the same samples,
        then there is a redundancy and one of the values can be dropped.
        In the current case, this function reduces the amount of features from 36 to 24, because for 6 different features
        we have a single binary column instead of a 3-column one-hot encoding.

        Parameters:
        -----------
        df_data : dataframe to transform

        Returns: transformed dataframe"""

        df_res=df_data.copy()

        for c in df_res.columns:
            if len(df_res[c].unique()) < 3 or is_numeric_dtype(df_res[c]):
                continue

            df_res.loc[df_res[c] == 'No internet service', c]='No'
            df_res.loc[df_res[c] == 'No phone service', c]='No'

        return df_res

    def fit(self, df_data):
        """Docstring:
        Learns which columns are binary and which are categorical

        Parameters:
        -----------
        df_data : dataframe to train from

        Returns: self"""

        df_data = self.__remove_value_redundancy(df_data)

        self.binary_cols = [c for c in df_data.columns if len(df_data[c].unique()) == 2]
        print(f"binary cols => {self.binary_cols}")

        self.category_cols = [c for c in df_data.columns if len(df_data[c].unique()) > 2 and not is_numeric_dtype(df_data[c])]
        print(f"category cols => {self.category_cols}")

        return self

    def transform(self, df_data):
        """Docstring:
        Transforms binary columns to numeric
        Performs one-hot encoding on categorical variables

        Parameters:
        -----------
        df_data : dataframe to transform

        Returns: transformed dataframe"""

        df_data = self.__remove_value_redundancy(df_data)

        # binary to numeric
        for c in self.binary_cols:
            df_data[c] = self.lbb.fit_transform(df_data[c])

        # categorical to one-hot
        df_data = pd.get_dummies(df_data,
                                 columns=self.category_cols)

        return df_data

    def fit_transform(self, df_data):
        """Docstring:
        Fits and transforms dataframe to match sklearn predictor format

        Parameters:
        -----------
        df_data : dataframe to learn from and transform

        Returns: transformed dataframe"""

        self.fit(df_data)
        return self.transform(df_data)

