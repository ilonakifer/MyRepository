
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

from TenurePredictorBase import TenurePredictorBase


class TenureClassifier (TenurePredictorBase):
    """Docstring:
    Class for predicting customer tenure using multiple classifiers
    The class allows to choose the number of classes to divide into
    """

    ###############################
    # parameters for classification
    ###############################
    cls_params = {'max_depth' : 3,
                  'objective' : 'binary:logistic',
                  'learning_rate': 0.2,
                  'min_child_weight': 1,
                  'n_estimators': 100,
                  'gamma': 0.001,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'scale_pos_weight': 1,
                  'scoring': 'logloss',}

    def __init__(self, **kwargs):
        """Docstring:
        Init function
        optional arguments -
        model_type : regression or classification
        num_classes : number of classes to divide into (onyl relevant for classification)
        """

        # gets model type arg
        self.model_type = 'classification'

        # gets num classes arg
        self.num_classes = 3
        if 'num_classes' in kwargs:
            self.num_classes = kwargs['num_classes']

        # instantiate model
        self.model_list = [XGBClassifier(**self.cls_params) for _ in range(self.num_classes)]

        # variable instantiation
        self.__bin2cls_dict = {}
        self.__cls_means = {}

    def __build_classes_from_bins(self, bins):
        """Docstring:
        turn buckets into classes

        Parameters:
        -----------
        bins : edges of buckets for defining classes
        """

        bins = [np.round(b*1000)/1000 for b in bins]
        bin2cls_dict = {}
        for i in range(len(bins)-1):

            cls = i
            bucket = pd.Interval(bins[i],
                                 bins[i+1], closed='right')

            bin2cls_dict[bucket] = cls

        return bin2cls_dict

    def __bin2class(self, bin_assignment):
        """Docstring:
        assign class by bucket

        Parameters:
        -----------
        bin_assignment : array of value buckets

        Returns: array of assigned classes
        """

        cls_assignment = [self.__bin2cls_dict[ba] for ba in bin_assignment]

        return cls_assignment

    def __calc_class_means(self, vals, cls_assignment):
        """Docstring:
        calculate mean per class, for assigning to classification predictions

        Parameters:
        -----------
        vals : array of actual tenure vals
        cls_assignment : array of class assignments

        Returns: dictionary of average tenure per class
        """

        cls_means = {}
        for cls in np.unique(cls_assignment):

            cls_vals = [vals[i] for i in range(len(cls_assignment)) if cls_assignment[i] == cls]
            cls_means[cls] = np.mean(cls_vals)

        return cls_means

    def fit(self, X, y):
        """Docstring:
        fit model given train data and label

        Parameters:
        -----------
        X : training samples
        y : labels
        Returns: self"""

        print(f"Dividing into {self.num_classes} bins:\n {list(self.__bin2cls_dict.keys())}")
        bin_assignment, bins = pd.cut(y, self.num_classes, retbins=True)

        self.__bin2cls_dict = self.__build_classes_from_bins(bins)

        cls_assignment = self.__bin2class(bin_assignment)
        self.__cls_means = self.__calc_class_means(y, cls_assignment)

        feature_importances = [[] for _ in range(len(self.__cls_means.keys()))]
        for cls in sorted(self.__cls_means.keys()):

            print(f"Running binary classifier for class {cls}")

            # Calculate weights of positive class vs. negative class
            scale_pos_weight = len([i for i in range(len(cls_assignment)) if cls_assignment[i] != cls]) / \
                               len([i for i in range(len(cls_assignment)) if cls_assignment[i] == cls])

            params = self.model_list[cls].get_params()
            params['scale_pos_weight'] = scale_pos_weight
            self.model_list[cls].set_params(**params)
            self.model_list[cls].fit(X,
                                     [1 if cls_assignment[i] == cls else 0 for i in range(len(cls_assignment))])
            feature_importances[cls] = self.model_list[cls].feature_importances_

        self.feature_importances_ = np.mean(feature_importances,
                                            axis=0)

        return self

    def predict(self, X):
        """Docstring:
        predict tenure for given samples

        Parameters:
        -----------
        X : samples to predict for
        Returns: prediction array"""

        pred_mat = [[] for _ in range(len(self.__cls_means.keys()))]
        for cls in self.__cls_means.keys():
            pred_mat[cls] = self.model_list[cls].predict_proba(X)[:,1]

        cls_predictions = np.argmax(np.array(pred_mat),
                                    axis=0)
        val_predictions = [self.__cls_means[cls] for cls in cls_predictions]

        return val_predictions



