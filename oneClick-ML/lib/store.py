import pathlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_mean_cm(df_test_metrics, res_dir_cl):

	confusion_matrices = df_test_metrics['confusion_matrix'].values
	confusion_matrices_array = np.array(confusion_matrices.tolist())
	mean_confusion_matrix = pd.DataFrame(np.mean(confusion_matrices_array, axis=0))
	std_confusion_matrix = pd.DataFrame(np.std(confusion_matrices_array, axis=0))

	plt.figure(figsize=(8, 6))
	ax = sns.heatmap(mean_confusion_matrix, annot=False, fmt=".2f", cmap="Blues", xticklabels=mean_confusion_matrix.columns, 
					 yticklabels=mean_confusion_matrix.index)
	
	for i in range(mean_confusion_matrix.shape[0]):
		for j in range(mean_confusion_matrix.shape[1]):
			mean_value = mean_confusion_matrix.iloc[i, j]
			std_value = std_confusion_matrix.iloc[i, j]
			ax.text(j + 0.5, i + 0.5, f'{mean_value:.2f} ± {std_value:.2f}', 
					ha='center', va='center', color='black', fontsize=10)

	plt.xlabel('Predicted labels')
	plt.ylabel('True labels')
	plt.title('Mean Confusion Matrix')
	plt.tight_layout()
	plt.savefig(res_dir_cl / 'mean_cm.png', dpi=400)
	plt.close()
    
def store_mean_metrics(best_train_metrics, df_test_metrics, res_dir_cl):
    
    # Validate expected columns are present
    expected_train_cols = ['mean_test_mcc', 'mean_test_roc_auc', 'mean_test_f1']
    missing = [col for col in expected_train_cols if col not in best_train_metrics.columns]
    if missing:
        raise ValueError(f"Missing expected train metric columns: {missing}")

    # Rename train metric columns to match test for consistency
    train_metrics_renamed = best_train_metrics.rename(columns={
        'mean_test_mcc': 'mcc',
        'mean_test_roc_auc': 'roc_auc',
        'mean_test_f1': 'f1'
    })

    # Drop confusion matrix column if present in test metrics
    df_test_clean = df_test_metrics.drop(columns=['confusion_matrix'], errors='ignore')

    # Calculate means and stds
    mean_metrics = pd.DataFrame([
        df_test_clean.mean(),
        df_test_clean.std(),
        train_metrics_renamed[['mcc', 'roc_auc', 'f1']].mean(),
        train_metrics_renamed[['mcc', 'roc_auc', 'f1']].std()
    ], index=['Test (mean)', 'Test (std)', 'Train (mean best)', 'Train (std)'])

    # Save to CSV
    round(mean_metrics, 2).to_csv(res_dir_cl / 'mean_metrics.csv')


def save_raw_results(X_test_idx_list, real_y_list, pred_y_list, pred_prob_list, res_dir_cl):
	# Flatten the lists
	X_test_flat = [item for sublist in X_test_idx_list for item in sublist]
	y_real_flat = [item for sublist in real_y_list for item in sublist]
	y_pred_flat = [item for sublist in pred_y_list for item in sublist]
	y_proba_flat = [item for sublist in pred_prob_list for item in sublist]

	# Create a DataFrame directly from the flattened lists
	df_raw_results = pd.DataFrame({
								'idx': X_test_flat,
								'real y': y_real_flat,
								'pred y': y_pred_flat,
								'prob y=1': np.round(y_proba_flat,2)})

	# add a column that tells us about real y=1 -> set 1 if real and pred are = 1 and 0 if they are different
	df_raw_results['true prediction y=1'] = np.where(df_raw_results['pred y'] == 1, df_raw_results['real y'], np.nan)
	
	
	# store DataFrame of raw results
	df_raw_results.to_csv(res_dir_cl / 'raw_results.csv', index=False)

def get_patient_prob_results(X_idx, X_test_idx_list, pred_prob_list, rs_list, res_dir_cl):
    import pandas as pd
    from collections import defaultdict

    # Collect probabilities per patient
    prob_dict = defaultdict(list)
    seen_dict = defaultdict(set)

    for col_name, (test_idxs, prob_y) in zip(rs_list, zip(X_test_idx_list, pred_prob_list)):
        for idx, y_prob in zip(test_idxs, prob_y):
            prob_dict[idx].append((col_name, round(y_prob, 2)))
            seen_dict[idx].add(col_name)

    # Build DataFrame with all X_idx entries
    df_probs = pd.DataFrame(index=X_idx)
    for rs in rs_list:
        df_probs[rs] = pd.NA  # Initialize with NaN

    for idx in prob_dict:
        for col_name, val in prob_dict[idx]:
            df_probs.loc[idx, col_name] = val

    df_probs.index.name = 'idx'
    df_probs.reset_index(inplace=True)

    # Save to CSV
    df_probs.to_csv(res_dir_cl / 'raw_res_per_patient.csv', index=False)

    
def store_classification_metrics(test_metrics_list, test_metrics, best_train_metrics_list, train_metrics, best_params_list,  res_dir_cl):

    df_test_metrics = pd.DataFrame(test_metrics_list, columns=test_metrics)
    df_test_metrics['confusion_matrix'] = np.array(df_test_metrics['confusion_matrix'])
    df_test_metrics.to_csv(res_dir_cl / 'test_metrics.csv')

    plot_mean_cm(df_test_metrics, res_dir_cl)

    pd.DataFrame(best_train_metrics_list, columns=train_metrics).round(2).to_csv(res_dir_cl / 'best_train_metrics.csv')
    pd.DataFrame(best_params_list).to_csv(res_dir_cl / 'best_params.csv')

    store_mean_metrics(pd.DataFrame(best_train_metrics_list, columns=train_metrics), df_test_metrics, res_dir_cl)

def mean_roc_curve_plot(tpr_list, roc_auc_list, fpr_common, res_dir_cl):
	tpr_array = np.array(tpr_list)
	roc_auc_array = np.array(roc_auc_list)

	# Compute the mean and standard deviation of TPR values
	mean_tpr = np.mean(tpr_array, axis=0)
	std_tpr = np.std(tpr_array, axis=0)

	# Compute the mean AUC
	mean_auc = np.mean(roc_auc_array)
	std_auc = np.std(roc_auc_array)

	# Plotting
	plt.figure(figsize=(12, 8))
	plt.plot(fpr_common, mean_tpr, color='blue', lw=2, label=f'Mean ROC curve (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

	# Shaded standard deviation with a different color
	plt.fill_between(fpr_common, mean_tpr - std_tpr, mean_tpr + std_tpr, color='blue', alpha=0.2, label='Standard deviation')

	# Dashed line for random guessing
	plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2, label='Random guess')

	# Grid lines
	#plt.grid(True, linestyle='--', linewidth=0.5)

	# Axis limits
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])

	# Labels and title
	plt.xlabel('False Positive Rate', fontsize=14)
	plt.ylabel('True Positive Rate', fontsize=14)
	plt.title('Mean ROC Curve with Standard Deviation', fontsize=16)

	# Legend
	plt.legend(loc='lower right', fontsize=12)

	# Display plot
	plt.savefig(res_dir_cl / 'roc_curve.png', dpi=400)