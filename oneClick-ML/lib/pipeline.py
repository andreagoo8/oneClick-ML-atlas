from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, confusion_matrix, f1_score, make_scorer, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC


import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu
from scipy.stats import mannwhitneyu, shapiro

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# Sklearn Preprocessing
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Sampling Strategies
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression



def define_pipeline(model_name, param_distribution, handle_imb_data, include_feature_selector, n_of_features, feature_types, random_state):
    """
    Returns a complete pipeline and a dictionary of parameters for randomized/grid search.

    Args:
        model_name (str): One of 'randomforestclassifier', 'extratreesclassifier', 
                          'xgbclassifier', 'logisticregression', 'svc'
        param_distribution (dict): Dict of parameter grids per model name + imbalance methods ('SMOTE', 'RUS').
        handle_imb_data (str): One of 'no', 'SMOTE', 'UnderSampling', 'all'
        include_feature_selector (bool): Include FeatureSelector step or not.
        n_of_features (int): Number of features to select (if enabled).
        feature_types (dict): Dict of feature types (e.g., {'numeric': [...], 'categorical': [...]}).

    Returns:
        tuple: (pipeline, param_grid)
    """
    model_classes = {
    'randomforestclassifier': lambda: RandomForestClassifier(random_state=random_state),
    'extratreesclassifier': lambda: ExtraTreesClassifier(random_state=random_state),
    'xgbclassifier': lambda: xgb.XGBClassifier(random_state=random_state),
    'logisticregression': lambda: LogisticRegression(random_state=random_state),
    'svc': lambda: SVC(probability=True, random_state=random_state)
    }


    # Initialize classifier or ensemble
    if model_name == 'voting':
        estimator_list = [
            ('randomforestclassifier', model_classes['randomforestclassifier']()),
            ('svc', model_classes['svc']()),
            ('logisticregression', model_classes['logisticregression']()),
            ('xgbclassifier', model_classes['xgbclassifier']())
        ]
        classifier = VotingClassifier(estimators=estimator_list, voting='soft')
        model_param_grid = {}
        for name, _ in estimator_list:
            model_param_grid.update({
                f'classifier__{name}__{param}': values
                for param, values in param_distribution[name].items()
            })
    elif model_name == 'stacking':
        base_estimators = [
            ('svc', model_classes['svc']()),
            ('randomforestclassifier', model_classes['randomforestclassifier']()),
            ('xgbclassifier', model_classes['xgbclassifier']())
        ]
        final_estimator = model_classes['logisticregression']()
        classifier = StackingClassifier(estimators=base_estimators, final_estimator=final_estimator)
        model_param_grid = {}      
        for name, _ in base_estimators:
            model_param_grid.update({
                f'classifier__{name}__{param}': values
                for param, values in param_distribution[name].items()
            })
        model_param_grid.update({
            f'classifier__final_estimator__{param}': values
            for param, values in param_distribution['logisticregression'].items()
        })

    elif model_name in model_classes:
        classifier = model_classes[model_name]()
        model_param_grid = {
            f"classifier__{param}": values
            for param, values in param_distribution[model_name].items()
        }

    else:
        raise ValueError(f"Unsupported model name: {model_name}")





    # Optional imbalance handling parameters
    if handle_imb_data in ['SMOTE', 'all']:
        model_param_grid.update({
            'smote__sampling_strategy': param_distribution['SMOTE']['sampling_strategy'],
            'smote__k_neighbors': param_distribution['SMOTE']['k_neighbors']
        })

    if handle_imb_data in ['UnderSampling', 'all']:
        model_param_grid.update({
            'rus__sampling_strategy': param_distribution['RUS']['sampling_strategy']
        })

    # Pipeline construction
    pipeline_steps = []

    if include_feature_selector:
        pipeline_steps.append(('feature_selector', FeatureSelector(alpha=0.05, top_k=n_of_features, feature_types=feature_types)))

    pipeline_steps.append(('dynamic_preprocessor',DynamicPreprocessor(full_feature_types=feature_types)))

    if handle_imb_data == 'SMOTE':
        pipeline_steps.append(('smote', SMOTE(random_state=random_state)))
    elif handle_imb_data == 'UnderSampling':
        pipeline_steps.append(('rus', RandomUnderSampler(random_state=random_state)))
    elif handle_imb_data == 'all':
        pipeline_steps.extend([
            ('smote', SMOTE(random_state=random_state)),
            ('rus', RandomUnderSampler(random_state=random_state))
        ])

    pipeline_steps.append(('classifier', classifier))

    pipeline = Pipeline(pipeline_steps)

    return pipeline, [model_param_grid]



class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to perform feature selection using appropriate statistical tests
    based on feature types (numerical, categorical, or binary).
    
    Parameters:
    -----------
    alpha : float, optional (default=0.05)
        The p-value threshold for feature selection.
    
    top_k : int, optional (default=None)
        The number of top features to keep based on smallest p-values.
    
    feature_types : dict, optional (default=None)
        A dictionary mapping feature names to their types ('numerical', 'categorical', 'binary').
       
    """
    def __init__(self, alpha=0.05, top_k=None, feature_types=None):
        self.alpha = alpha
        self.top_k = top_k
        self.feature_types = feature_types  # Dictionary mapping features to types
        self.selected_features_ = None  # Store selected feature names
    
    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples!"

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        
        feature_names = X.columns
        p_values = []
        
        feature_mapping = {}
        for feature_type, feature_list in (self.feature_types or {}).items():
            for feature in feature_list:
                feature_mapping[feature] = feature_type
        

        # Compute p-values based on feature types
        for col in feature_names:
            feature_type = feature_mapping.get(col, 'numerical')  # Default to numerical
            
            if feature_type == 'numerical':  # t-test
                group0 = X.loc[y[y == 0].index, col]
                group1 = X.loc[y[y == 1].index, col]
				
				# check for zero variance 
                if group0.nunique() <= 1 or group1.nunique() <= 1:
                   # Not enough variation for a parametric test -> use Mann-Whitney U test
                    _, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
                else:
					# Perform normality test (Shapiro-Wilk) - Shapiro-Wilk is reliable for small to medium sample sizes.
                    normal0 = shapiro(group0.dropna()).pvalue > 0.05
                    normal1 = shapiro(group1.dropna()).pvalue > 0.05

                    if normal0 and normal1:
						# Both groups are normally distributed -> use t-test
                        _, p_value = ttest_ind(group0, group1, equal_var=True, nan_policy='omit')
                    else:
						# At least one group is not normally distributed -> use Mann-Whitney U test
                        _, p_value = mannwhitneyu(group0, group1, alternative='two-sided')

            elif feature_type == 'ordinal':  # Mann-Whitney U test (non-parametric)
                group0 = X.loc[y[y == 0].index, col]
                group1 = X.loc[y[y == 1].index, col]
                _, p_value = mannwhitneyu(group0, group1, alternative='two-sided')

            elif feature_type in ['nominal', 'binary']:  # Chi-square test
                contingency_table = pd.crosstab(X[col], y)
                _, p_value, _, _ = chi2_contingency(contingency_table)

            else:
                raise ValueError(f"Unsupported feature type: {feature_type} for feature {col}")
            
            p_values.append((col, p_value))
            
        p_values_df = pd.DataFrame(p_values, columns=['Feature', 'P-Value']).dropna()
        
        # Select features based on method
        if self.top_k:
            self.selected_features_ = p_values_df.nsmallest(self.top_k, 'P-Value')['Feature'].tolist()
        else:
            self.selected_features_ = p_values_df[p_values_df['P-Value'] < self.alpha]['Feature'].tolist()
        
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.selected_features_]
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_features_) if self.selected_features_ else np.array([])

class DynamicPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, full_feature_types):
        self.full_feature_types = full_feature_types
        self.column_transformer = None

    def fit(self, X, y=None):
        selected_cols = X.columns
        selected_feature_types = self._filter_feature_types(selected_cols)

        # Define transformers
        ordinal_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OrdinalEncoder())
        nominal_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist'))
        numerical_transformer = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        binary_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'))

        transformers = []
        if 'ordinal' in selected_feature_types:
            transformers.append(('ordinal', ordinal_transformer, selected_feature_types['ordinal']))
        if 'nominal' in selected_feature_types:
            transformers.append(('nominal', nominal_transformer, selected_feature_types['nominal']))
        if 'numerical' in selected_feature_types:
            transformers.append(('numerical', numerical_transformer, selected_feature_types['numerical']))
        if 'binary' in selected_feature_types:
            transformers.append(('binary', binary_transformer, selected_feature_types['binary']))

        self.column_transformer = ColumnTransformer(transformers)
        self.column_transformer.fit(X, y)
        return self

    def transform(self, X):
        return self.column_transformer.transform(X)

    def _filter_feature_types(self, selected_columns):
        selected_feature_types = {}
        for ftype, features in self.full_feature_types.items():
            selected = list(set(features) & set(selected_columns))
            if selected:
                selected_feature_types[ftype] = selected
        return selected_feature_types


    def get_feature_names_out(self, input_features=None):
        # Return feature names after transformation (e.g., after one-hot encoding)
        feature_names = []

        for name, trans, cols in self.column_transformer.transformers_:
            if hasattr(trans, 'get_feature_names_out'):
                names = trans.get_feature_names_out(cols)
            elif isinstance(trans, Pipeline):
                last_step = trans.steps[-1][1]
                if hasattr(last_step, 'get_feature_names_out'):
                    names = last_step.get_feature_names_out(cols)
                else:
                    names = cols
            else:
                names = cols
            feature_names.extend(names)

        return np.array(feature_names)

def get_score():
	score ={'roc_auc': 'roc_auc',
		'accuracy': 'accuracy',
		'f1': 'f1',
		'recall': 'recall',
		'mcc': make_scorer(matthews_corrcoef, greater_is_better = True),
		'auprc': make_scorer(average_precision_score)}
	return score

def get_final_transformed_test_data(grid_model, X_test):
    """
    Extracts the final transformed test set and corresponding feature names
    after feature selection and preprocessing.

    Parameters:
    -----------
    grid_model : fitted GridSearchCV or pipeline
    X_test : pd.DataFrame
    classifier_name : str
        E.g., 'LR', 'SVC', 'XGB', etc.
    classifier_names : dict or list
        Mapping or list of classifier names for pipeline lookup

    Returns:
    --------
    X_test_transformed : np.ndarray
        Transformed test data (ready for prediction or SHAP).
    final_colnames : list of str
        Final feature names after full pipeline processing.
    model : trained classifier object
    """

    # Auto-named steps in make_pipeline
    selector = grid_model.best_estimator_.named_steps.get('feature_selector', None)
    preprocessor = grid_model.best_estimator_.named_steps['dynamic_preprocessor']
    

    # Step 1: Apply feature selection if applicable
    if selector:
        X_selected = selector.transform(X_test)
        selected_features = selector.get_feature_names_out()
        X_selected = pd.DataFrame(X_selected, columns=selected_features)
    else:
        X_selected = X_test

    # Step 2: Apply preprocessing
    X_test_transformed = preprocessor.transform(X_selected)
    final_colnames = preprocessor.get_feature_names_out().tolist()

    return X_test_transformed, final_colnames

def evaluate_model(grid_model, X_test, y_test, fpr_common):
    y_pred = grid_model.predict(X_test)
    y_pred_proba = grid_model.predict_proba(X_test)[:, 1]

    metrics = {
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=grid_model.classes_),
        'mcc': round(matthews_corrcoef(y_test, y_pred), 2),
        'f1': round(f1_score(y_test, y_pred), 2),
        'roc_auc': round(roc_auc_score(y_test, y_pred_proba), 2),
        'auprc': round(average_precision_score(y_test, y_pred_proba), 2)
    }

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    tpr_interp = np.interp(fpr_common, fpr, tpr)
    roc_auc = auc(fpr, tpr)

    return y_pred, y_pred_proba, metrics, tpr_interp, roc_auc

def my_grid_search(X, y, pipeline, param_dist, n_jobs, refit_metric, scoring, cv_repeats, cv_splits, n_iter, random_state,verbose=True):
	''' Grid Search Definition | Function Definition
		This function instatiate cv object, gridsearchcv object and fits it. It returns the fitted grid_search and the dataframe cv_results_
		X : train data
		y : train labels
		param_grid : (list) dictionary of parameters that will be tested in the grid search. 
		param_dist : list dictionary of parameters (as distribution) used in randomizedsearchcv
		refit : string, default: 'mcc'
		scoring : dicitionary of scores that will be used in grid search
		cv_repeats: how many time cross validation will be repeated. default: 5
		cv_splits: number of folds. default: 5
	'''
	cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=random_state)

	grid_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=cv, n_jobs=n_jobs, verbose=verbose, refit=refit_metric, scoring=scoring, random_state=random_state, n_iter=n_iter)

	grid_search.fit(X, y)

    
	return grid_search