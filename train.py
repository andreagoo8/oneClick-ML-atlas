import argparse
import pathlib
import random
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import train_test_split
from lib.store import get_patient_prob_results, mean_roc_curve_plot, save_raw_results, store_classification_metrics
from lib.importance import compute_shap_values, shap_anaysis, store_importances
from lib.utils import create_result_dirs, generate_paths, load_config, load_data, load_param_distributions, save_config
from lib.pipeline import define_pipeline, evaluate_model, get_final_transformed_test_data, get_score, my_grid_search

from os import path, makedirs, listdir
import os


# --- ADDED from walter ---
file_basepath = "local_data"

def train(
    # --- general ---
    project,
    di,
    y_label = "ylabel",
    col_to_drop = "FutureMotorFluctuations FutureFreezing FutureFalls max_status_longi CognitiveStatus",
    include_feature_selector = False,
    n_of_features = 17,
    classifiers_list = "randomforestclassifier extratreesclassifier xgbclassifier logisticregression svc",
    shap_analysis = True,
    perm_importance = False,

    # --- gridsearch params ---
    n_jobs = 30,
    refit = "mcc",
    cv_repeats = 5,
    cv_splits = 3,
    n_iter = 20,
    train_test_split_size = 0.2,
    handle_imb_data = "no",
    nested_iterations = 30,

    # --- new ---
    target_model_name = "best_model",
    path_to_feature_types = "local_data/feature_list_types"
    ):

    # Input/output folders
    if (not path.exists(file_basepath)):
        os.mkdir(file_basepath)

    data_dir = file_basepath + 'data'
    output_dir = file_basepath + 'results'
    
    
    if (not path.exists(data_dir)):
        os.mkdir(data_dir)
    if (not path.exists(output_dir)):
        os.mkdir(output_dir)              
    
    data_dir = di.download(data_dir);



    # reproducibility settings
    random_state = 42
    random.seed(0)

    # load config dictionary from config path
    # config = load_config(config_file)
    
    # load parameter distributions for gridsearch
    param_distributions = load_param_distributions('params.yaml')
    
    print(f'Parameter distributions loaded: {param_distributions}')

    # get input dir and create res dir (Pathlib dir)
    input_dir, res_dir = generate_paths(y_label, data_dir, output_dir)
    
    # store config options for this run
    # save_config(config, res_dir)

    # rs list for train-test splits
    rs_list = random.sample(range(1, 100), nested_iterations)

    # load data
    X, y, feature_types, strat_col = load_data(input_dir, path_to_feature_types,
                                   y_label=y_label, col_to_drop=col_to_drop , stratify_on_symptom=True) # false if analysis of non_active patients


    # metrics definition (train and test metrics)
    test_metrics = ['mcc', 'f1', 'roc_auc', 'confusion_matrix', 'auprc']
    train_metrics = ['mean_test_mcc', 'std_test_mcc', 'mean_test_roc_auc', 'std_test_roc_auc', 'mean_test_f1', 'std_test_f1']

    # classifiers loop
    for classifier in classifiers_list.split():

        res_dir_cl, feat_dir, shap_dir = create_result_dirs(res_dir, classifier, shap_analysis, perm_importance)
        print('-----------------------------------')
        print(f'Running gridsearch on {classifier}')  
        

        # pipeline definition
        pipeline, param_dist = define_pipeline(classifier, param_distribution=param_distributions, 
                                               handle_imb_data = handle_imb_data,
                                               include_feature_selector = include_feature_selector, 
                                               n_of_features = n_of_features, 
                                               feature_types = feature_types, random_state=random_state)



        print('Pipeline : ', pipeline)
        print('--')
        print('Param dist: ', param_dist)

        # init storage
        test_metrics_list, best_train_metrics_list, best_params_list = [], [], []
        tpr_list, fpr_list, roc_auc_list = [], [], []
        shap_val_list, shap_data_list, perm_imp_list = [], [], []
        pred_prob_list, pred_y_list, real_y_list, X_test_idx_list = [], [], [], []
        fpr_common = np.linspace(0, 1, 100)

        # train-test combinations loop
        for rs in rs_list:

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=strat_col, test_size=train_test_split_size, random_state=rs)
            
            
            # run grid search
            grid_model = my_grid_search(X_train, y_train, pipeline, param_dist, n_jobs, verbose=True, 
                                        refit_metric=refit, scoring=get_score(), 
                                        cv_repeats=cv_repeats, cv_splits=cv_splits, n_iter=n_iter, random_state=random_state)

            X_test_idx_list.append(X_test.index)
            X_test_transformed, final_colnames = get_final_transformed_test_data(grid_model, X_test)
            print('final_colnames: ', final_colnames)
            # test best model
            y_pred, y_pred_proba, metrics, tpr_interp, roc_auc = evaluate_model(grid_model, X_test, y_test, fpr_common)

            # append results 
            pred_y_list.append(y_pred)
            pred_prob_list.append(y_pred_proba)
            real_y_list.append(y_test)
            test_metrics_list.append(metrics)
            tpr_list.append(tpr_interp)
            fpr_list.append(fpr_common)
            roc_auc_list.append(roc_auc)
            best_params_list.append(grid_model.best_params_)

            best_train_metrics_list.append([
                grid_model.cv_results_.get(key)[grid_model.best_index_]
                for key in train_metrics
            ])

            # shap analysis
            if shap_analysis:
                df_shap, df_data = compute_shap_values(classifier, grid_model.best_estimator_.named_steps['classifier'], X_test_transformed, final_colnames, rs, res_dir=shap_dir)
                shap_val_list.append(df_shap)
                shap_data_list.append(df_data)

            # permutation importance analysis
            if perm_importance:
                perm = permutation_importance(grid_model, X_train, y_train, n_repeats=10, random_state=random_state,
                                        scoring=make_scorer(matthews_corrcoef, greater_is_better=True))
                perm_imp_list.append(np.mean(perm.importances, axis=1))

        
        print(f'Saving metrics in {res_dir_cl}')
        # feature importances saving
        if perm_importance:
            store_importances(feat_dir, perm_imp_list, feature_names_orig=X.columns)
        if shap_analysis:
            shap_anaysis(shap_dir, shap_val_list, shap_data_list)
        
        # Save performances
        store_classification_metrics(test_metrics_list, test_metrics, best_train_metrics_list, train_metrics, best_params_list, res_dir_cl)
        
        # store raw results
        save_raw_results(X_test_idx_list, real_y_list, pred_y_list, pred_prob_list, res_dir_cl)
        get_patient_prob_results(X.index, X_test_idx_list, pred_prob_list, rs_list, res_dir_cl)
        
        # plot mean cm
        mean_roc_curve_plot(tpr_list, roc_auc_list, fpr_common, res_dir_cl)

        # important adding!!!!
        project.log_model(name=target_model_name, kind="model", source=str(output_dir))       
       
    return 0


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run analysis with optional configuration.")
	
	parser.add_argument("--config", type=str, help="Path to the configuration file")
	
	args = parser.parse_args()
	
	sys.exit(main(config_file=args.config))
