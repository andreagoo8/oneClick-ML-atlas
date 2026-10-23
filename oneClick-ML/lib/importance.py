

import pathlib
from pydoc import pathdirs
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_values(classifier_name, model, X_test_transformed, colnames, rs, res_dir):
    #step_model = model.best_estimator_.named_steps[classifier_name]
    
    if classifier_name in ['logisticregression', 'svc']:
        explainer = shap.PermutationExplainer(model.predict_proba, X_test_transformed)
    else:
        explainer = shap.TreeExplainer(model)
    
    if classifier_name == 'xgbclassifier':
        explanation = explainer(X_test_transformed) # extract shap values for class label 1 if XGB
    else:	
        explanation = explainer(X_test_transformed)[:,:,1] # extract shap values for class label 1 if RF or other classifiers

    df_vals = pd.DataFrame(explanation.values, columns=colnames)
    df_data = pd.DataFrame(explanation.data, columns=colnames)
    
    explanation.feature_names = colnames

    plt.figure()
    shap.summary_plot(explanation, show=False)
    plt.tight_layout()
    single_shap = pathlib.Path(res_dir / 'single_shaps')
    single_shap.mkdir(parents=True, exist_ok=True)

    plt.savefig(single_shap / f'summary_{rs}.png', dpi=400)
    plt.close()

    return df_vals, df_data



def store_importances( res_dir_feat, perm_imp_list, feature_names_orig):

	# df of permutation importance values for each iteration
	df_perm_imp = pd.DataFrame(perm_imp_list, columns=feature_names_orig)
	df_perm_imp.to_csv(res_dir_feat / 'all_permutation_importances.csv', index = False)


def shap_anaysis(shap_dir, shap_val_list, shap_data_list):

	df_shap_values = pd.concat([df for df in shap_val_list], axis=0)
	df_shap_values.to_csv(shap_dir / 'df_shap_vals_filtered.csv', index=False)

	df_shap_data = pd.concat([df for df in shap_data_list], axis=0)
	df_shap_data.to_csv(shap_dir / 'df_shap_data_filtered.csv', index=False)
	

	num_rows = 500 if len(df_shap_values) > 700 else len(df_shap_values)  # Adjust as needed
	random_indices = np.random.choice(len(df_shap_values), size=num_rows, replace=False)

	# Sample the same rows from both DataFrames
	df_shap_vals = df_shap_values.iloc[random_indices]
	df_shap_data = df_shap_data.iloc[random_indices]

	shap_values_obj = shap.Explanation(values=df_shap_vals.values, data=df_shap_data, 
									feature_names=df_shap_data.columns) # BASE VALUES kept as default..


	# swarmplot
	plt.figure()
	shap.plots.beeswarm(shap_values_obj, max_display=10, show=False)
	plt.tight_layout()
	plt.savefig(shap_dir / 'beeswarmplot.png', dpi=400)
	plt.close()

	plt.figure()
	shap.summary_plot(shap_values_obj, show=False)
	plt.tight_layout()
	plt.savefig(shap_dir / 'summary_plot.png', dpi=400)
	plt.close()
