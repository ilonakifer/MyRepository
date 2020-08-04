
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from TenureClassifier import TenureClassifier
from TenureRegressor import TenureRegressor

from DataTransformer import DataTransformer

def main():
    """Docstring:
    Main program for tenure prediction
    Running usage: predict_tenure.py <data-file-name> <'classification' or 'regression'> <num_classes>

    Parameters:
        1. data-file-name: csv file containing features and labels
        2. model-type: can be either classification or regression. Default: regression
        3. num-classes: if classification, number of classes to divide into. Ignored for regression
    """

    ##############################
    # Get program arguments
    ##############################
    if len(sys.argv) <= 1:
        print(f"Usage: {sys.argv[0]} <data-file> <classification-type> <num_classes>")
        sys.exit(1)

    # data filename arg
    data_filename = sys.argv[1]

    # model type arg
    model_type = "regression"
    if len(sys.argv) > 2:
        model_type = sys.argv[2]

    # number of classes arg
    num_classes = 3
    if len(sys.argv) > 3:
        num_classes=int(sys.argv[3])

    ##############################
    # read data file
    ##############################
    df_data = pd.read_csv(data_filename,
                          index_col=0).drop(columns=['TotalCharges'])
    ##############################
    # Transform into learnable format
    ##############################
    target = 'tenure'

    dt = DataTransformer()
    df_transformed_data = dt.fit_transform(df_data)

    #############################
    # Split into train/test
    #############################
    feature_cols = [c for c in df_transformed_data.columns if c not in [target , 'Churn']]
    X_train, X_test, y_train, y_test = train_test_split(df_transformed_data[feature_cols].values,
                                                        df_transformed_data[target].values,
                                                        test_size=0.33,
                                                        stratify=df_transformed_data.Churn,
                                                        random_state=23)
    ##############################
    # Run predictions
    ##############################

    if model_type == "regression":
        cp_model = TenureRegressor()
    else:
        cp_model = TenureClassifier(num_classes=num_classes)

    cp_model.fit(X_train, y_train)

    ##############################
    ## Feature importances
    ##############################
    importances = cp_model.feature_importances_

    indices = np.argsort(importances)[::-1]
    features = [df_transformed_data.columns[i] for i in range(len(importances)) if importances[i] > 0]

    print(f"Most important out of  {len(features)} features:")
    for f in range(10):
        print(f"\t{f+1} feature {df_transformed_data.columns[indices[f]]} importance {importances[indices[f]]:.2f}")

    ##############################
    # predict test and score
    ##############################
    score = cp_model.score(X_test,
                           y_test)
    print(f"Mean Absolute error score on test: {score:.2f}")

##############################
# Run program
##############################
if __name__ == "__main__":
    main()