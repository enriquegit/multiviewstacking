import pkg_resources
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_example_data():
    """Returns an example dataset with two views.
       The example data is taken from the HTAD database
       A Home-Tasks Activities Dataset with Wrist-accelerometer
       and Audio Features [1].

       The dataset contains wrist-accelerometer and audio data from people 
       performing at-home tasks such as sweeping, brushing teeth, washing hands,
       or watching TV. The first set of features (v1_*) are from audio signals
       and the second set of features (v2_*) were extracted from an accelerometer.

    Returns
    -------
    tuple : A tuple with 7 elements containing the example dataset.
        (X_train, y_train, X_test, y_test, ind_v1, ind_v2, le)
        X_train: training data features.
        y_train: training data labels (encoded as integers).
        X_test: test data features (encoded as integers).
        y_test: test data labels.
        ind_v1: the column indices for view 1 features.
        ind_v2: the column indices for view 2 features.
        le: The LabelEncoder object used to encode classes into integers.
            This can be used to restore the original class labels.
    
        
    References
    ----------

    .. [1] Garcia-Ceja, Enrique, et al. "Htad: A home-tasks activities dataset with
       wrist-accelerometer and audio features."
       MultiMedia Modeling: 27th International Conference, MMM 2021.
    
    """


    stream = pkg_resources.resource_stream(__name__, 'data/htad.csv')

    # Read file.
    #df = pd.read_csv('htad.csv')
    df = pd.read_csv(stream, encoding='latin-1')

    # Encode labels into integers.
    le = LabelEncoder()
    df["class"] = le.fit_transform(df["class"])

    # Store class in y.
    y = df["class"]

    # Store features in x.
    X = df.drop(["class"], axis = 1)

    # Get column names.
    colnames = list(X.columns)

    # Get column indices for each view.
    ind_v1 = [colnames.index(x) for x in colnames if "v1_" in x]
    ind_v2 = [colnames.index(x) for x in colnames if "v2_" in x]

    random_seed = 200

    # Split into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size = 0.5,
                                                        stratify = y, 
                                                        random_state = random_seed)
    
    return (X_train, y_train, X_test, y_test, ind_v1, ind_v2, le)
