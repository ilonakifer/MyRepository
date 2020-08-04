
from sklearn.metrics import mean_absolute_error


class TenurePredictorBase:
    """Docstring:
    Base Class for predicting customer tenure.
    """
    model = None
    model_type = ''

    def __init__(self, **kwargs):
        """Docstring:
        Init function
        """

        # variable instantiation
        self.feature_importances_ = []

    def fit(self, X, y):
        """Docstring:
        PlaceHolder that should be implementing in inheriting classes
        """
        pass

    def predict(self, X):
        """Docstring:
        PlaceHolder that should be implementing in inheriting classes
        """
        pass

    def feature_importances_(self):
        """Docstring:
        Returns feature importances
        If regression then a single list of feature importances
        If classification then a list of feature importances per class
        """
        return self.feature_importances_


    def score(self, X, true):
        """Docstring:
        Performs evaluation of actual labels vs. prediction.
        Behaviour depends on type of model -
        1. For regression, evaluation is on actual tenure value using MAE
        2. For classification, true labels are transformed to class number, and evaluation is via f1 score

        Parameters:
        -----------
        true : array of actual labels
        pred : array of predicted labels
        Returns: prediction KPI value"""

        pred = self.predict(X)
        return mean_absolute_error(true,
                                   pred)





