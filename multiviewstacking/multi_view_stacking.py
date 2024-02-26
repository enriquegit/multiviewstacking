import pandas as pd
import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

class MultiViewStacking(BaseEstimator, ClassifierMixin):
    """A Multi-View Stacking classifier.

    An implementation of the Multi-View Stacking algorithm Garcia-Ceja, E. et al. [1]
    which is based on Stacked Generalization proposed by Wolpert, D. H. [3]. 

    Parameters
    ----------
    views_indices : a list of tuples or a list of lists, default=None
        The list of tuples specifies the column start/end indices for each view as a range.
        The list of tuples has the form [(start_v1,end_v1),(start_v2,end_v2),...(start_vn,end_vn)]
        where start_v1 is the column index where the features of the first view begin
        and end_v1 is the index where the features of the first view end and so on.
        
        If the features for a given view are no contiguous,
        it is also possible to explicitly specify the column indices
        using a list of lists of the form [[],[],...,[]].
        The inner lists contain the column indices that belong to a given view.
        If there are three views then the list should contain three sublists.
                
        If this paremeter is not specified, all features will be mapped into a single view.
        
    first_level_learners : a list of scikit-learn classifiers, default=None.
        The list must have n elements where n is the number of views.
        Each element can be a classifier from the scikit-learn library.
        It also supports custom models as long as they implement the
        fit(), predict(), and predict_proba() methods and return similar
        values as those in scikit-learn.
        The classifier at position i is trained with the column indices
        specified at position i in views_indices. If this parameter is None,
        a RandomForestClassifier(random_state=123) will be used for each of the views.
    
    meta_learner : a scikit-learn classifier, default=RandomForestClassifier(random_state=123).
    
    k : an int, default = 10. 
        The number of folds to be used during internal k-fold cross validation.
    
    random_state : int, default=123
        random state used for the internal k-fold cross validation.


    Attributes
    ----------
    num_views_ : an integer indicating the number of detected views based
        on the size of views_indices.
    
    fitted_first_level_learners_ : the list of first-level-learners after
        they have been trained with the input data.

    References
    ----------

    .. [1] Garcia-Ceja, E., Galván-Tejada, C. E., & Brena, R. "Multi-view stacking for activity
           recognition with sound and accelerometer data", Information Fusion, (2018).
    
    .. [2] Wolpert, D. H., "Stacked generalization", Neural networks, 5(2), 241-259, (1992).
    
    .. [3] Multi-View Stacking in R
           https://enriquegit.github.io/behavior-free/ensemble.html#stacked-generalization
    
    """
    
    def __init__(self, 
                 views_indices = None,
                 first_level_learners = None,
                 meta_learner = RandomForestClassifier(random_state=123),
                 k = 10,
                 random_state = 123):
        
        self.views_indices = views_indices
        self.first_level_learners = first_level_learners
        self.meta_learner = meta_learner
        self.k = k
        self.random_state = random_state
    
    def __validate_parameters(self, X):
        """Validate parameters passed in __init__.
           
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
            
        """
                
        # Validate views_indices.
        if self.views_indices is None or len(self.views_indices) == 0:
            # If no indices were supplied, create a single view with all features.
            warnings.warn("No views_indices were specified. All features will be mapped into a single view.")
            self.views_indices = [[*range(0, X.shape[1], 1)]]
        else:
            # Expand range indices (start, end) into [start, ...., end]
            if type(self.views_indices[0]) is tuple:
                self.views_indices = [[*range(x[0], (x[1]+1), 1)] for x in self.views_indices] 
        
        # Check if there are intersections with the indices and show a warning if so.
        flat_list = [
            x
            for xs in self.views_indices
            for x in xs
        ]
        # Get duplicate indices.
        dups = pd.Series(flat_list)[pd.Series(flat_list).duplicated()].values
        if len(dups) > 0:
            warnings.warn("One or more column indices appear in more than one viw. The following column index/indices appear in more than one view: " + str(dups))
        
        # Validate that the number of indices lists is equal to the number of first-level-learners.
        nviews = len(self.views_indices)
        if self.first_level_learners is None:
            # Set Random Forest as default learner but raise a warning.
            warnings.warn("No first-level-learners were defined. Using Random Forest as default.")
            self.first_level_learners = [RandomForestClassifier(random_state=123) for i in range(nviews)]
        else:
            if nviews != len(self.first_level_learners):
                raise Exception("The number of views differs from the number of first_level_learners.")
        
    
    def fit(self, X, y):
        """Build a Multi-View Stacking classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels) as integers. If target
            values are strings, you can use LabelEncoder() to
            convert them into integers.

        Returns
        -------
        self : MultiViewStacking
            Fitted estimator.
        """
        
        
        # Check that X and y have correct shape.
        X, y = check_X_y(X, y)
        
        # If X is a data frame, convert it to numpy.
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Validate parameters.
        self.__validate_parameters(X)
        
        # Set the number of views.
        self.num_views_ = len(self.views_indices)
        
        # Save unique labels as strings.
        self.uniquelabels_ = np.unique(y)
        self.uniquelabels_ = [str(x) for x in self.uniquelabels_ ]
        
        self.classes_ = unique_labels(y)
        
        # Save number of instances.
        self.num_rows = len(y)
        
        # List to save the true labels.
        truelabels = []
        
        # List of lists to save views' predictions.
        preds = [[] for x in range(self.num_views_)]
        
        # List of lists to save views' scores.
        scores = [None] * self.num_views_
        
        # Perform k-fold cross validation.
        kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.random_state)

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            
            xtrain, xtest = X[train_index], X[test_index]
            ytrain, ytest = y[train_index], y[test_index]

            # Iterate views.
            for j in range(self.num_views_):
                
                # Select columns for a specific view based on the provided indices.
                v_train = xtrain[:, self.views_indices[j]]
                v_test = xtest[:, self.views_indices[j]]

                # Fit corresponding model for view j.
                m_v = self.first_level_learners[j].fit(v_train, ytrain)

                # Predict labels.
                labels_v = m_v.predict(v_test)
                preds[j].append(labels_v)
                
                # Predict scores.
                raw_v = m_v.predict_proba(v_test)
                
                if scores[j] is not None:
                    scores[j] = np.vstack((scores[j], raw_v))
                else:
                    scores[j] = raw_v

            #### End training individual views ####
            
            # Append true labels.
            truelabels.append(ytest)
    
        #### End of cross validation ####

        # Flatten the lists.
        truelabels = np.concatenate(truelabels).ravel()
        
        for v in range(self.num_views_):
            preds[v] = np.concatenate(preds[v]).ravel()
        
        # Fit first-level learners again but with all data so no instances are wasted.
        for v in range(self.num_views_):
            self.first_level_learners[v].fit(X[:, self.views_indices[v]], y)
        
        
        # Make the trained base learners accessible.
        self.fitted_first_level_learners_ = self.first_level_learners
        
        # Construct meta-features.

        # Average scores
        avgscores = np.zeros(scores[0].shape, dtype=np.float64)
        
        for v in range(self.num_views_):
            avgscores = avgscores + scores[v]
        
        avgscores = avgscores / self.num_views_
        
        # Convert scores to data frame.
        df = pd.DataFrame(avgscores)
        
        # Convert column names to strings.
        df.columns = df.columns.astype(str)

        # Add predictions' columns to data frame.
        pred_cols = {}
        for v in range(self.num_views_):
            pred_cols['preds_v'+str(v+1)] = preds[v]
        
        df = df.assign(**pred_cols)
        
        # One-hot encode the predictions.
        categories = []
        for v in range(self.num_views_):
            categories.append(self.uniquelabels_)

        self.enc_ = OneHotEncoder(categories=categories,
                                  drop='first',
                                  sparse_output=False).set_output(transform="pandas")

        
        predcolnames = ['preds_v'+str(x+1) for x in range(self.num_views_)]
        predcols = df[predcolnames]

        ohe_predcols = self.enc_.fit_transform(predcols)

        # Remove original prediction columns and concatenate one-hot-encoded columns.
        metaX = pd.concat([df, ohe_predcols],axis=1).drop(columns=predcolnames)
        metaY = truelabels

        # Train metalearner.
        self.meta_learner.fit(metaX, metaY)
        
        # Return the classifier
        return self
    
    def __predict_labels_scores(self, X, predict_labels=True):
        """Auxiliary function to predict class labels or scores of the input samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
            
        predict_labels : boolean, default_True
            if True, return the labels predicted by the underlying meta-learner.
            if False, return the class scores predicted by the underlying meta-learner.

        Returns
        -------
        predictions : the classes or scores returned by the underlying meta-learner.
        
        """
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Input validation
        X = check_array(X)
        
        # List of lists to save views' predictions.
        preds = [[] for x in range(self.num_views_)]
                
        # List of lists to save views' scores.
        scores = [None] * self.num_views_
        
        # Iterate views.
        for j in range(self.num_views_):
            
            v_test = X[:, self.views_indices[j]]
            preds_v = self.first_level_learners[j].predict(v_test)
            preds[j].append(preds_v)
                
            scores_v = self.first_level_learners[j].predict_proba(v_test)
            if scores[j] is not None:
                scores[j] = np.vstack((scores[j], scores_v))
            else:
                scores[j] = scores_v
        
        
        # Average scores
        avgscores = np.zeros(scores[0].shape, dtype=np.float64)
        
        for v in range(self.num_views_):
            avgscores = avgscores + scores[v]
        
        avgscores = avgscores / self.num_views_
        
        df = pd.DataFrame(avgscores)

        # Change column names to strings.
        df.columns = df.columns.astype(str)

        # Add predictions columns.
        pred_cols = {}
        for v in range(self.num_views_):
            pred_cols['preds_v'+str(v+1)] = preds[v][0]
        
        df = df.assign(**pred_cols)
        
        predcolnames = ['preds_v'+str(x+1) for x in range(self.num_views_)]
        predcols = df[predcolnames]

        ohe_predcols = self.enc_.fit_transform(predcols)

        # Remove preds cols and concatenate encoded cols.
        metaX = pd.concat([df, ohe_predcols],axis=1).drop(columns=predcolnames)

        if predict_labels == True:
            predictions = self.meta_learner.predict(metaX)
        else:
            predictions = self.meta_learner.predict_proba(metaX)
        
        return predictions
        
    
    def predict(self, X):
        """Predict class labels of the input samples X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        predictions : the classes returned by the underlying meta-learner.
            Typically, a ndarray of shape (n_samples,).
        """
        
        # Check if fit has been called
        check_is_fitted(self)
        
        return self.__predict_labels_scores(X, predict_labels=True)
    
    def predict_proba(self, X):
        """Predict class scores of the input samples X.

        The predicted class scores are the ones predicted by the meta-learner.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        predictions : the scores returned by the underlying meta-learner.
            Typically, a ndarray of shape (n_samples, n_classes).
            The order of the classes corresponds to that in the
            attribute :term:`classes_`.
        """
        
        # Check if fit has been called
        check_is_fitted(self)
        
        return self.__predict_labels_scores(X, predict_labels=False)
    
    