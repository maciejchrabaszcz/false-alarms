import os
from typing import Tuple, Union, Optional, List

import pandas as pd
from jsonargparse import ActionConfigFile, ArgumentParser
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

CATEGORICAL_FEATURES = [
    "categoryname",
    "ipcategory_name",
    "ipcategory_scope",
    "parent_category",
    "grandparent_category",
    "start_hour",
    "start_minute",
    "start_second",
    "weekday",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "n10",
    "isiptrusted",
    "enforcementscore",
    "dstipcategory_dominate",
    "srcipcategory_dominate",
    "dstportcategory_dominate",
    "srcportcategory_dominate",
    "ip_first",
    "ip_second",
    "ip_third",
    "ip_fourth",
    "p6",
    "p9",
    "p5m",
    "p5w",
    "p5d",
    "p8m",
    "p8w",
    "p8d",
]
NUMERIC_FEATURES = [
    "overallseverity",
    "timestamp_dist",
    "correlatedcount",
    "score",
    "srcip_cd",
    "dstip_cd",
    "srcport_cd",
    "dstport_cd",
    "alerttype_cd",
    "direction_cd",
    "eventname_cd",
    "severity_cd",
    "reportingdevice_cd",
    "devicetype_cd",
    "devicevendor_cd",
    "domain_cd",
    "protocol_cd",
    "username_cd",
    "srcipcategory_cd",
    "dstipcategory_cd",
    "untrustscore",
    "flowscore",
    "trustscore",
    "thrcnt_month",
    "thrcnt_week",
    "thrcnt_day",
]


def load_train_data(
    path: str = "data/cybersecurity_training_preprocessed.csv",
    columns_to_drop: List[str] = None,
) -> pd.DataFrame:
    """Loads train data from path.

    Args:
        path (str, optional): Path to csv file. Defaults to "data/cybersecurity_training_preprocessed.csv".

    Returns:
        pd.DataFrame: training pandas dataframe.
    """
    df = pd.read_csv(path, sep="|")
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype("str")
    if columns_to_drop is not None:
        df = df.drop(columns_to_drop, axis=1)
    return df


def load_test_data(
    path: str = "data/cybersecurity_test_preprocessed.csv",
    columns_to_drop: List[str] = None,
) -> pd.DataFrame:
    """Loads test data from path.

    Args:
        path (str, optional): Path to csv file. Defaults to "data/cybersecurity_test_preprocessed.csv".

    Returns:
        pd.DataFrame: test pandas dataframe.
    """
    df = pd.read_csv(path, sep="|")
    if columns_to_drop is not None:
        df = df.drop(columns_to_drop, axis=1)
    return df


def train_holdout_split(
    df: pd.DataFrame, random_state: int = 123
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into train and holdout sets.

    Args:
        df (pd.DataFrame): dataframe to be splitted
        random_state (int, optional): random state. Defaults to 123.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train and holdout sets.
    """
    train_df, holdout_df = train_test_split(df, random_state=random_state)
    return train_df, holdout_df


def prepare_experiment(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    normalize: bool = True,
    normalize_method: str = "zscore",
    rare_to_value: float = 0.001,
    rare_value: str = "other",
    max_encoding_ohe: int = 25,
    log_experiment: bool = True,
    log_plots: Union[bool, list] = True,
    fold: int = 5,
    fix_imbalance: bool = False,
    fix_imbalance_method: str = "SMOTE",
    low_variance_threshold: float = 0.0,
    remove_multicollinearity: bool = True,
    remove_outliers: bool = True,
    use_gpu: bool = False,
) -> ClassificationExperiment:
    """Prepare PyCaret experiment.

    Args:
        train_df (pd.DataFrame): training dataframe.
        holdout_df (pd.DataFrame): holdout dataframe.
        normalize (bool, optional): normalize data. Defaults to True.
        rare_to_value (float, optional): rare to value. Defaults to 0.001.
        rare_value (str, optional): rare value. Defaults to "other".
        max_encoding_ohe (int, optional): max encoding ohe. Defaults to 25.
        log_experiment (bool, optional): log experiment. Defaults to True.
        log_plots (bool, optional): log plots. Defaults to True.
        fold (int, optional): fold. Defaults to 5.
        fix_imbalance (bool, optional): fix imbalance. Defaults to False.
        remove_multicollinearity (bool, optional): whether to remove multicollinearity. Defaults to True.
        remove_outliers (bool, optional): remove outliers. Defaults to True.

    Returns:
        ClassificationExperiment: PyCaret experiment.
    """
    experiment = ClassificationExperiment()
    experiment.setup(
        data=train_df,
        test_data=holdout_df,
        target="notified",
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        ignore_features=["client_code", "alert_ids"],
        normalize=normalize,
        normalize_method=normalize_method,
        rare_to_value=rare_to_value,
        rare_value=rare_value,
        max_encoding_ohe=max_encoding_ohe,
        log_experiment=log_experiment,
        log_plots=log_plots,
        fold=fold,
        fix_imbalance=fix_imbalance,
        fix_imbalance_method=fix_imbalance_method,
        low_variance_threshold=low_variance_threshold,
        remove_multicollinearity=remove_multicollinearity,
        remove_outliers=remove_outliers,
        use_gpu=use_gpu,
        index=False,
        html=False,
    )
    experiment.add_metric(
        "pr_auc", "PR AUC", average_precision_score, target="pred_proba"
    )
    return experiment


def get_best_models(
    experiment: ClassificationExperiment,
    n: int = 5,
    sort_metric="AUC",
    include_models: Optional[List[str]] = None,
):
    """Get best models.

    Args:
        experiment (ClassificationExperiment): PyCaret experiment.
        n (int, optional): number of models. Defaults to 5.
        sort_metric (str, optional): metric with respect to which
            best models are selected. Defaults to "AUC".

    Returns:
        [type]: Best models with respect to sort_metric.
    """
    return experiment.compare_models(
        sort=sort_metric, n_select=n, include=include_models
    )


def tune_models(
    experiment: ClassificationExperiment,
    models,
    optimize: str = "AUC",
    n_iter=10,
):
    """Tune models.

    Args:
        experiment (ClassificationExperiment): PyCaret experiment.
        models ([type]): models to be tuned.
        optimize (str, optional): metric with respect to which
            models are tuned. Defaults to "AUC".
        n_iter (int, optional): number of iterations. Defaults to 10.

    Returns:
        [type]: tuned models.
    """
    if isinstance(models, list):
        return [
            experiment.tune_model(
                model,
                optimize=optimize,
                n_iter=n_iter,
                search_library="optuna",
                early_stopping=True,
                tuner_verbose=10,
            )
            for model in models
        ]
    return experiment.tune_model(
        models,
        optimize=optimize,
        n_iter=n_iter,
        search_library="optuna",
        early_stopping=True,
        tuner_verbose=10,
    )


def main() -> None:
    global CATEGORICAL_FEATURES, NUMERIC_FEATURES
    """Trains and evaluates model on cybersecurity dataset."""
    parser = ArgumentParser()
    parser.add_argument(
        "--data.train_file",
        type=str,
        help="Path to train file",
        default="data/cybersecurity_training_preprocessed.csv",
    )
    parser.add_argument(
        "--data.test_file",
        type=str,
        help="Path to test file",
        default="data/cybersecurity_test_preprocessed.csv",
    )
    parser.add_argument(
        "--data.columns_to_drop",
        type=list,
        help="Columns to drop",
        default=["alert_ids", "client_code"],
    )
    parser.add_function_arguments(
        prepare_experiment, "experiment", skip=["train_df", "holdout_df"]
    )
    parser.add_function_arguments(
        get_best_models, "models_creation", skip=["experiment"]
    )
    parser.add_function_arguments(
        tune_models, "models_tuning", skip=["experiment", "models"]
    )
    parser.add_argument(
        "--blend_models",
        action="store_true",
        help="Flag saying whether to blend top models.",
    )
    parser.add_argument(
        "--tune_all_best_models",
        action="store_true",
        help="Flag saying whether to tune all top models.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="results",
        help="path to folder to save models and predictions",
    )
    parser.add_argument("-c", "--config", action=ActionConfigFile)
    config = parser.parse_args()
    CATEGORICAL_FEATURES = [
        x for x in CATEGORICAL_FEATURES if x not in config.data.columns_to_drop
    ]
    NUMERIC_FEATURES = [
        x for x in NUMERIC_FEATURES if x not in config.data.columns_to_drop
    ]
    train_df = load_train_data(
        config.data.train_file, config.data.columns_to_drop
    )
    train_df, holdout_df = train_holdout_split(train_df)
    experiment = prepare_experiment(train_df, holdout_df, **config.experiment)
    best_model = get_best_models(experiment, **config.models_creation)

    if config.tune_all_best_models or config.blend_models:
        tuned_model = tune_models(
            experiment, best_model, **config.models_tuning
        )
    else:
        tuned_model = tune_models(
            experiment, best_model[0], **config.models_tuning
        )
    if config.blend_models:
        tuned_model = experiment.blend_models(tuned_model)
    else:
        tuned_model = tuned_model[0]
    finalized_model = experiment.finalize_model(tuned_model)
    os.makedirs(config.save_path, exist_ok=True)
    experiment.save_model(tuned_model, config.save_path + "tuned_model")
    experiment.save_model(
        finalized_model, config.save_path + "finalized_model"
    )
    test_df = load_test_data(
        config.data.test_file, config.data.columns_to_drop
    )
    test_preds = experiment.predict_model(
        finalized_model, data=test_df, raw_score=True
    )["prediction_score_1"]
    with open(config.save_path + "test_preds.txt", "w") as f:
        f.write("\n".join([str(x) for x in test_preds]))

    experiment.save_experiment(config.save_path + "experiment")


if __name__ == "__main__":
    main()
