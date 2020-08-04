
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
        """
        return self.feature_importances_


    def score(self, X, true):
        """Docstring:
        Performs prediction and then scores the actual labels vs. prediction.
        Predictor accuracy is measure via MAE (Mean Absolute Error).
        This metric provides good interpretability of results, as well as
        equal weighting of errors.

        Parameters:
        -----------
        X : array of actual labels
        pred : array of predicted labels
        Returns: prediction KPI value"""

        pred = self.predict(X)
        return mean_absolute_error(true,
                                   pred)





