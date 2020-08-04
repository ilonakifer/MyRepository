
from xgboost import XGBRegressor

from TenurePredictorBase import TenurePredictorBase


class TenureRegressor (TenurePredictorBase):
    """Docstring:
    Class for predicting customer tenure using Regression
    """

    ###############################
    # parameters for regression
    ###############################
    reg_params = {'max_depth': 4,
                  'objective': 'reg:squarederror',
                  'learning_rate': 0.1,
                  'min_child_weight': 1,
                  'n_estimators': 100,
                  'gamma': 0.001,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'scale_pos_weight': 1}

    def __init__(self, **kwargs):
        """Docstring:
        Init function
        optional arguments -
        model_type : regression or classification
        num_classes : number of classes to divide into (onyl relevant for classification)
        """
        self.model_type = 'regression'
        self.model = XGBRegressor(**self.reg_params)

    def fit(self, X, y):
        """Docstring:
        fit model given train data and label

        Parameters:
        -----------
        X : training samples
        y : labels
        Returns: self"""

        self.model.fit(X, y)
        self.feature_importances_= self.model.feature_importances_

        return self

    def predict(self, X):
        """Docstring:
        predict tenure for given samples

        Parameters:
        -----------
        X : samples to predict for
        Returns: prediction array"""
        return self.model.predict(X)




