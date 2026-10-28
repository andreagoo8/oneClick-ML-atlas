import argparse
import pathlib
import random
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

def main(config_file)->int:

    # reproducibility settings
    random_state = 42
    random.seed(0)


    # load config dictionary from config path
    config = load_config(config_file)

    # load parameter distributions for gridsearch
    param_distributions = load_param_distributions('params.yaml')

    # get input dir and create res dir (Pathlib dir)
    input_dir, res_dir = generate_paths(config['y_label'], config['input_path'], config['output_path'])

    # store config options for this run
    save_config(config, res_dir)

    # rs list for train-test splits
    rs_list = random.sample(range(1, 100), config['nested_iterations'])

    # load data
    X, y, feature_types, strat_col = load_data(input_dir, config['path_to_feature_types'],
                                   y_label=config['y_label'], col_to_drop=config['col_to_drop'], stratify_on_symptom=True) # false if analysis of non_active patients


    # metrics definition (train and test metrics)
    test_metrics = ['mcc', 'f1', 'roc_auc', 'confusion_matrix', 'auprc']
    train_metrics = ['mean_test_mcc', 'std_test_mcc', 'mean_test_roc_auc', 'std_test_roc_auc', 'mean_test_f1', 'std_test_f1']

    # classifiers loop
    for classifier in config['classifiers_list']:

        res_dir_cl, feat_dir, shap_dir = create_result_dirs(res_dir, classifier, config['shap_analysis'], config['perm_importance'])
        print('-----------------------------------')
        print(f'Running gridsearch on {classifier}')


        # pipeline definition
        pipeline, param_dist = define_pipeline(classifier, param_distribution=param_distributions,
                                               handle_imb_data = config['handle_imb_data'],
                                               include_feature_selector = config['include_feature_selector'],
                                               n_of_features = config['n_of_features'],
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

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=strat_col, test_size=config['train_test_split_size'], random_state=rs)


            # run grid search
            grid_model = my_grid_search(X_train, y_train, pipeline, param_dist, config['n_jobs'], verbose=True,
                                        refit_metric=config['refit'], scoring=get_score(),
                                        cv_repeats=config['cv_repeats'], cv_splits=config['cv_splits'], n_iter=config['n_iter'], random_state=random_state)

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
            if config['shap_analysis']:
                df_shap, df_data = compute_shap_values(classifier, grid_model.best_estimator_.named_steps['classifier'], X_test_transformed, final_colnames, rs, res_dir=shap_dir)
                shap_val_list.append(df_shap)
                shap_data_list.append(df_data)

            # permutation importance analysis
            if config['perm_importance']:
                perm = permutation_importance(grid_model, X_train, y_train, n_repeats=10, random_state=random_state,
                                        scoring=make_scorer(matthews_corrcoef, greater_is_better=True))
                perm_imp_list.append(np.mean(perm.importances, axis=1))


        print(f'Saving metrics in {res_dir_cl}')
        # feature importances saving
        if config['perm_importance']:
            store_importances(feat_dir, perm_imp_list, feature_names_orig=X.columns)
        if config['shap_analysis']:
            shap_anaysis(shap_dir, shap_val_list, shap_data_list)

        # Save performances
        store_classification_metrics(test_metrics_list, test_metrics, best_train_metrics_list, train_metrics, best_params_list, res_dir_cl)

        # store raw results
        save_raw_results(X_test_idx_list, real_y_list, pred_y_list, pred_prob_list, res_dir_cl)
        get_patient_prob_results(X.index, X_test_idx_list, pred_prob_list, rs_list, res_dir_cl)

        # plot mean cm
        mean_roc_curve_plot(tpr_list, roc_auc_list, fpr_common, res_dir_cl)



    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis with optional configuration.")

    parser.add_argument("--config", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    sys.exit(main(config_file=args.config))

### --- LINES ADDED FOR THE USE OF ATLAS --- ###

import os
from os import path

file_basepath = "local_data"

def helper_main(
    project,

# Place here the name of the data and config objects that has been created and fetched in the notebook
    csv_file,
    config_file,
    output_artifact_name
    ):

    try:

# Create a Path object for "local_data", and if that directory doesn't already exist, create it
        base_dir = pathlib.Path(file_basepath)
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)

# Define three subdirectory paths under base_dir, and create them if they don't already exist
        data_dir = base_dir / 'data'
        config_dir = base_dir / 'config'
        output_dir = base_dir / 'results'


        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)

# Use the 'di' object to download data into data_dir (converted to a string path), overwriting existing files if present
        data_dir = pathlib.Path(csv_file.download(str(data_dir), overwrite=True))

# Use the 'config_file' object to download data into data_dir (converted to a string path), overwriting existing files if present
        config_path = pathlib.Path(config_file.download(str(config_dir), overwrite=True))

# Calls the original main function
        main(config_file=str(config_path))

# Places the outputs in a digitalhub model
        project.log_artifact(name=output_artifact_name, kind="model", source=str(output_dir))

    except Exception as e:
        print(f"Error during training: {e}")
