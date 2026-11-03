import configparser
import pathlib
from random import randint
from typing import Dict, List, Optional, Tuple
import warnings
import pandas as pd
from scipy.stats import randint, uniform, loguniform
import yaml


def load_config(config_file):
    dict_config = {}

    if config_file:
        print("----------------------------------------")
        print(f"Using configuration file: {config_file}")

        config = configparser.ConfigParser()
        config.read(config_file)

        try:
            # Accessing general values
            dict_config['y_label'] = config.get('general', 'y_label')
            dict_config['col_to_drop'] = config.get('general', 'col_to_drop').split()
            dict_config['include_feature_selector'] = config.get('general', 'include_feature_selector')
            if dict_config['include_feature_selector'] == 'False': 
               dict_config['include_feature_selector'] = False
               dict_config['n_of_features'] = 'all'

            dict_config['n_of_features'] = config.getint('general','n_of_features')
            dict_config['path_to_feature_types'] = config.get('general', 'path_to_feature_types')
            
            dict_config['classifiers_list'] = config.get('general', 'classifiers_list').split()
            dict_config['shap_analysis'] = config.get('general', 'shap_analysis')
            if dict_config['shap_analysis'] == 'False': 
                dict_config['shap_analysis']=False
            dict_config['perm_importance'] = config.get('general', 'perm_importance')
            if dict_config['perm_importance'] == 'False': 
                dict_config['perm_importance']=False

			# Path settings
            dict_config['input_path'] = config.get('general', 'input_path', fallback='../data/X_neuroart.csv')
            dict_config['output_path'] = config.get('general', 'output_path', fallback='../results/all_input_patients_w_bl_var/')
			
            # Accessing GridSearch parameters
            
            dict_config['n_jobs'] = config.getint('gridsearch params', 'n_jobs', fallback=6)
            dict_config['refit'] = config.get('gridsearch params', 'refit', fallback='mcc')
            dict_config['cv_repeats'] = config.getint('gridsearch params', 'cv_repeats', fallback=3)
            dict_config['cv_splits'] = config.getint('gridsearch params', 'cv_splits', fallback=5)
            dict_config['n_iter'] = config.getint('gridsearch params', 'n_iter', fallback=50)
            dict_config['train_test_split_size'] = config.getfloat('gridsearch params', 'train_test_split_size', fallback=0.2)
            dict_config['handle_imb_data'] = config.get('gridsearch params', 'handle_imb_data', fallback='no')
            dict_config['nested_iterations'] = config.getint('gridsearch params', 'nested_iterations', fallback=30)

        except KeyError as e:
            print(f"Missing key in the configuration file: {e}")
            return 1

    return dict_config

def load_param_distributions(yaml_path):
    print('-------------------------------')
    print('Loading Parameter Distributions')
    with open(yaml_path, 'r') as file:
        yaml_content = file.read()
    # Define custom constructors for different tags

    def range_constructor(loader, node):
        values = loader.construct_sequence(node)
        return range(*values)

    def uniform_constructor(loader, node):
        values = loader.construct_sequence(node)
        return uniform(loc=values[0], scale=values[1] - values[0])

    def randint_constructor(loader, node):
        values = loader.construct_sequence(node)
        return randint(low=values[0], high=values[1] + 1)

    def loguniform_constructor(loader, node):
        values = loader.construct_sequence(node)
        return loguniform(values[0], values[1])

    # Register the constructors with the yaml loader
    yaml.add_constructor('!python/range', range_constructor)
    yaml.add_constructor('!uniform', uniform_constructor)
    yaml.add_constructor('!randint', randint_constructor)
    yaml.add_constructor("!loguniform", loguniform_constructor)

    # Load the YAML content
    data = yaml.load(yaml_content, Loader=yaml.FullLoader)
    print("---------------------------------")
    
    return data



def generate_paths(y_label, input_path, output_path):

    input_dir = pathlib.Path(input_path)

    # make dir if it does not exist (result dir)
    res_dir = pathlib.Path(output_path) / f'{y_label}'
    res_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, res_dir

def save_config(config, res_dir):
	
	with open(res_dir / "config.txt", "w") as file:
		# General values
		file.write("General values:\n")
		file.write(f"    - Y Label: {config['y_label']}\n")
		file.write(f"    - Columns to drop: {config['col_to_drop']}\n")
		file.write(f"    - Path to feature types list: {config['path_to_feature_types']}\n")
		file.write(f"	 - output path: {config['output_path']}\n")
		file.write(f"    - input path: {config['input_path']}\n")
		file.write(f"    - Feature selection: {config['include_feature_selector']}\n")
		if config['include_feature_selector']:
			file.write(f"    - N. of features to keep: {config['n_of_features']}\n")
		file.write(f"	 - Shap analyisis: {config['shap_analysis']}\n")
		# GridSearch parameters
		file.write("GridSearch parameters:\n")
		file.write(f"    - n_jobs: {config['n_jobs']}\n")
		file.write(f"    - refit: {config['refit']}\n")
		file.write(f"    - cv_repeats: {config['cv_repeats']}\n")
		file.write(f"    - cv_splits: {config['cv_splits']}\n")
		file.write(f"    - n_iter: {config['n_iter']}\n")
		file.write(f"    - train_test_split_size: {config['train_test_split_size']}\n")
		file.write(f"    - handle_imb_data: {config['handle_imb_data']}\n")

def create_result_dirs(base_dir, classifier_name, shap_analysis, perm_importance):
	
    base = pathlib.Path(base_dir) / classifier_name
    feat_dir =  base / "feature_analysis"
    shap_dir = feat_dir / "shap_analysis"


    for path in [base, feat_dir, shap_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return base, feat_dir, shap_dir


def load_data(
    input_path: str,
    input_feature_lists: str,
    y_label: str,
    col_to_drop: Optional[List[str]] = None,
    stratify_on_symptom: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]], pd.Series]:
    """
    Loads data from a CSV file, preprocesses it, and categorizes features.

    Args:
        input_path (str): Path to the input CSV file containing the data.
        input_feature_lists (str): Path to the directory containing CSV files
                                   (e.g., nominal_features.csv, numerical_features.csv,
                                   ordinal_features.csv, binary_features.csv)
                                   that define feature categories.
        y_label (str): The name of the target column in the input data.
        col_to_drop (Optional[List[str]]): A list of additional columns to drop from X.
                                            Defaults to None.
        stratify_on_symptom (bool): If True, creates a stratification column
                                    based on a related symptom and the y_label.
                                    Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]], pd.Series]: A tuple containing:
            - X (pd.DataFrame): The preprocessed feature DataFrame.
            - y (pd.Series): The target variable Series.
            - feature_types (Dict[str, List[str]]): A dictionary categorizing features by type.
            - strat_col (pd.Series): The column to use for stratification.

    Raises:
        FileNotFoundError: If the input CSV file or 'y.csv' (if applicable) is not found.
        ValueError: If y_label is not found in the DataFrame and 'y.csv' is also not found.
    """
    print('Data Loading')
    try:
        X = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input data file not found at: '{input_path}'")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input data file is empty: '{input_path}'")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading '{input_path}': {e}")

    # --- Handle default columns to drop ---
    default_to_drop = ['PatientID', 'Center']
    present_default_cols = [col for col in default_to_drop if col in X.columns]
    if present_default_cols:
        X.drop(columns=present_default_cols, inplace=True)
        print(f'Columns {present_default_cols} removed from X.')
    else:
        print(f'Columns {default_to_drop} not present in the provided DataFrame.')

    # --- Extract target variable (y) and drop from X ---
    y: pd.Series
    if y_label in X.columns:
        y = X[y_label]
        X.drop(columns=[y_label], inplace=True)
        print(f"Target column '{y_label}' extracted and removed from X.")
    else:
        # Attempt to load y from a separate 'y.csv' if y_label not in X
        y_csv_path = f'{input_path}/y.csv'
        try:
            y = pd.read_csv(y_csv_path, header=None).squeeze("columns")
            print(f"Target column '{y_label}' not found in main DataFrame. Loaded 'y' from '{y_csv_path}'.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Target column '{y_label}' not found in the DataFrame, and "
                f"'{y_csv_path}' not found."
            )
        except pd.errors.EmptyDataError:
            raise ValueError(f"'{y_csv_path}' is empty. Cannot load target variable.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading '{y_csv_path}': {e}")

    # --- Drop other unwanted columns specified by col_to_drop ---
    if col_to_drop:
        cols_to_drop_present = [col for col in col_to_drop if col in X.columns]
        missing_cols = [col for col in col_to_drop if col not in X.columns]

        if missing_cols:
            warnings.warn(
                f"The following columns specified in 'col_to_drop' are not present "
                f"in the DataFrame and will not be dropped: {missing_cols}"
            )
        
        if cols_to_drop_present:
            X.drop(columns=cols_to_drop_present, inplace=True)
            print(f"Columns {cols_to_drop_present} removed from X.")
        else:
            print("No specified columns from 'col_to_drop' were present in X to remove.")
    else:
        print("No additional columns specified to drop.")

    # --- Determine stratification column (MUST be done BEFORE symptom-related feature is dropped from X) ---
    strat_col: Optional[pd.Series] = None
    if stratify_on_symptom:
        symptom_map = {
            'FutureMotorFluctuations': 'MotorFluctuations',
            'FutureFreezing': 'Freezing',
            'FutureDyskinesia': 'Dyskinesia'
        }
        sympt_for_strat = symptom_map.get(y_label)

        if sympt_for_strat and sympt_for_strat in X.columns:
            # Create stratification column by combining symptom and y_label
            strat_col = X[sympt_for_strat].astype(str) + "_" + y.astype(str) # Use y here as it's already extracted
            print(f"Stratification column created using '{sympt_for_strat}' and '{y_label}'.")
        elif sympt_for_strat:
            warnings.warn(
                f"Stratification symptom '{sympt_for_strat}' (derived from '{y_label}') "
                f"not found in DataFrame. Stratification will default to y_label."
            )
        else:
            warnings.warn(
                f"No specific symptom defined for y_label '{y_label}' for stratification. "
                f"Stratification will default to y_label."
            )
    
    # If strat_col was not set by stratify_on_symptom (e.g., symptom not found or disabled), use y
    if strat_col is None:
        strat_col = y
        print("Stratification column defaulted to the target variable 'y'.")

    # --- Explicitly drop symptom-related binary features from X if they exist ---
    # This ensures they are removed AFTER strat_col creation and BEFORE feature categorization logic or re-categorization fallback
    symptom_binary_features_to_remove_from_X = {
        'FutureMotorFluctuations': 'MotorFluctuations',
        'FutureDyskinesia': 'Dyskinesia',
        'FutureFreezing': 'Freezing'
    }
    
    symptom_to_drop_from_X = symptom_binary_features_to_remove_from_X.get(y_label)
    if symptom_to_drop_from_X and symptom_to_drop_from_X in X.columns:
        X.drop(columns=[symptom_to_drop_from_X], inplace=True)
        print(f"Explicitly removed symptom-related feature '{symptom_to_drop_from_X}' from X.")

    # --- Load feature categories from CSV files ---
    nominal_features: List[str] = []
    numerical_features: List[str] = []
    ordinal_features: List[str] = []
    binary_features: List[str] = []

    feature_category_files = {
        'nominal': f'{input_feature_lists}/nominal_features.csv',
        'numerical': f'{input_feature_lists}/numerical_features.csv',
        'ordinal': f'{input_feature_lists}/ordinal_features.csv',
        'binary': f'{input_feature_lists}/binary_features.csv'
    }

    for category, file_path in feature_category_files.items():
        try:
            features = pd.read_csv(file_path, header=None).squeeze("columns").tolist()
            if category == 'nominal':
                nominal_features = features
            elif category == 'numerical':
                numerical_features = features
            elif category == 'ordinal':
                ordinal_features = features
            elif category == 'binary':
                binary_features = features
        except FileNotFoundError:
            warnings.warn(
                f"Feature list file not found: '{file_path}'. "
                f"Assuming no features for '{category}' category."
            )
        except pd.errors.EmptyDataError:
            warnings.warn(
                f"Feature list file is empty: '{file_path}'. "
                f"Assuming no features for '{category}' category."
            )
        except Exception as e:
            warnings.warn(
                f"Error loading feature list from '{file_path}': {e}. "
                f"Assuming no features for '{category}' category."
            )

    # --- Remove symptom-related binary features from the loaded lists ---
    # This is important for the `feature_types` dictionary and `ordered_cols`
    symptom_binary_features_to_remove_from_list = {
        'FutureMotorFluctuations': 'MotorFluctuations',
        'FutureDyskinesia': 'Dyskinesia',
        'FutureFreezing': 'Freezing'
    }
    
    symptom_to_remove_from_list = symptom_binary_features_to_remove_from_list.get(y_label)
    if symptom_to_remove_from_list and symptom_to_remove_from_list in binary_features:
        binary_features.remove(symptom_to_remove_from_list)
        print(f"Removed '{symptom_to_remove_from_list}' from binary_features list as it's related to the target '{y_label}'.")
    elif symptom_to_remove_from_list:
        warnings.warn(
            f"Attempted to remove '{symptom_to_remove_from_list}' from binary_features list "
            f"for y_label '{y_label}', but it was not found in the list."
        )

    # --- Validate and re-categorize features if mismatch occurs ---
    total_categorized_features = (
        len(nominal_features) + len(numerical_features) +
        len(ordinal_features) + len(binary_features)
    )

    print(f'\n--- Feature Categorization Summary ---')
    print(f'Total specified attributes in categories: {total_categorized_features}')
    print(f'Nominal: {len(nominal_features)}, Numerical: {len(numerical_features)}, '
          f'Ordinal: {len(ordinal_features)}, Binary: {len(binary_features)}')
    print(f'Number of columns in X: {len(X.columns)}') # X.columns here should NOT contain the symptom feature
    print('------------------------------------')

    if total_categorized_features != len(X.columns):
        warnings.warn(
            "Mismatch between categorized features count and actual DataFrame columns count. "
            "All features will be treated as numerical to proceed."
        )
        numerical_features = X.columns.tolist()
        binary_features, nominal_features, ordinal_features = [], [], []
        print("Features re-categorized: All features are now treated as numerical.")

    # --- Ensure all categorized features are present in X and reorder X ---
    all_categorized_features_set = set(
        nominal_features + numerical_features + ordinal_features + binary_features
    )
    
    missing_from_X = all_categorized_features_set - set(X.columns)
    if missing_from_X:
        warnings.warn(
            f"The following categorized features are not found in X and will be excluded: "
            f"{list(missing_from_X)}"
        )
        nominal_features = [f for f in nominal_features if f in X.columns]
        numerical_features = [f for f in numerical_features if f in X.columns]
        ordinal_features = [f for f in ordinal_features if f in X.columns]
        binary_features = [f for f in binary_features if f in X.columns]

    ordered_cols = nominal_features + numerical_features + ordinal_features + binary_features
    
    X = X[ordered_cols]
    print(f"DataFrame X reordered to match feature categories. X now has {len(X.columns)} columns.")

    # --- Final feature types dictionary ---
    feature_types = {
        'nominal': nominal_features,
        'numerical': numerical_features,
        'ordinal': ordinal_features,
        'binary': binary_features
    }

    print('Data loaded successfully.')
    return X, y, feature_types, strat_col




'''

def load_data_(input_path, input_feature_lists, y_label, col_to_drop, stratify_on_symptom=True):
	print('Data Loading')
	X = pd.read_csv(input_path)

	
	default_to_drop = ['PatientID', 'Center']
	present_default_cols = [col for col in default_to_drop if col in X.columns]
	if present_default_cols:
		X.drop(columns=present_default_cols, inplace=True)
		print(f' Columns {present_default_cols} removed from X.')

	else:
		print(f' Columns {default_to_drop} not present in the provided Dataframe')


	strat_col = None
	if stratify_on_symptom:
		if y_label == 'FutureMotorFluctuations': sympt = 'MotorFluctuations'
		elif y_label == 'FutureFreezing': sympt = 'Freezing'
		else: sympt = 'Dyskinesia'
		
		strat_col = X[sympt].astype(str) + "_" + X[y_label].astype(str)
	
	# Check if y is a valid column name in DataFrame X
	try:
		if y_label in X.columns:
			y = X[y_label]
			X.drop(columns=[y_label], inplace=True)    
		else:
			try:
				y = pd.read_csv(input_path + '/y.csv').values.ravel()
			except FileNotFoundError:
				raise NameError(f" Invalid column name {y_label}, and 'y.csv' not found at {input_path}")
	except NameError as e:
		print(e)




	if col_to_drop:
		# drop the other unwanted columns (e.g. other y) in X (if present)
		# check if any col in col_to_drop is missing
		missing_cols = [col for col in col_to_drop if col not in X.columns]

		if missing_cols:
			warnings.warn(f" The following columns are not present in the DataFrame and will not be dropped: {missing_cols}")
			X.drop(columns=[col for col in col_to_drop if col in X.columns], inplace=True)
			print(f" Columns {set(col_to_drop) - set(missing_cols)} removed from X.")
		else:
			X.drop(columns=col_to_drop, inplace=True)
			print(f" Columns {col_to_drop} removed from X.")


	else:
		print(" No specified columns to drop.")

	nominal_features, numerical_features, ordinal_features, binary_features = [], [], [], []
	
	# search for csv files that specify features categories in the input data folder. If not present all features are treated as binary features (check preprocessing phase)
	try:
		nominal_features = pd.read_csv(f'{input_feature_lists}/nominal_features.csv', header=None).squeeze("columns").tolist()
	except FileNotFoundError:
		pass
	try:
		numerical_features = pd.read_csv(f'{input_feature_lists}/numerical_features.csv', header=None).squeeze("columns").tolist()
	except FileNotFoundError:
		pass
	try:
		ordinal_features = pd.read_csv(f'{input_feature_lists}/ordinal_features.csv', header=None).squeeze("columns").tolist()
	except FileNotFoundError:
		pass
	try:
		binary_features = pd.read_csv(f'{input_feature_lists}/binary_features.csv', header=None).squeeze("columns").tolist()
	except FileNotFoundError:
		pass
	
	# Drop binary features dyskinesia, motorfluctuations or freezing depending on the y_label from binary_features list:
	if y_label == 'FutureMotorFluctuations':
		binary_features.remove('MotorFluctuations')
	if y_label == 'FutureDyskinesia':
		binary_features.remove('Dyskinesia')
	if y_label == 'FutureFreezing':
		binary_features.remove('Freezing')

	# Check if the combined length of all categorized features matches the number of columns in X
	total_categorized_features = sum([
		len(nominal_features), 
		len(numerical_features), 
		len(ordinal_features), 
		len(binary_features)
	])
	
	print(f'total specified attributes in categories: {total_categorized_features}')
	print(f'{len(nominal_features)}, {len(numerical_features)}, {len(ordinal_features)}, {len(binary_features)}')
	print(f'number of columns in X: {len(X.columns)}')
	print('--')
	if total_categorized_features != len(X.columns):
		# Assume all features are numerical if they are not fully categorized
		numerical_features = X.columns.tolist()
		binary_features, nominal_features, ordinal_features = [], [], []
	print('Data loaded successfully.')


	ordered_cols = nominal_features + numerical_features + ordinal_features + binary_features
	X = X[ordered_cols]
	feature_types = {'nominal': nominal_features, 'numerical':numerical_features, 'ordinal':ordinal_features, 'binary':binary_features}


	# if strat_col is not provided, use y as stratification column
	strat_col = y if strat_col is None else strat_col

	
	return X, y, feature_types, strat_col


'''
