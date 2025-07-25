import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from imblearn.metrics import specificity_score, sensitivity_score


def compute_results(
    i_fold: int,
    model: str,
    prob: np.ndarray,
    y: np.ndarray,
    result: List[List[Union[int, str, float]]],
    rs: bool = False,
    rpn: int = 0,
    ths_range: Union[float, List[float]] = 0.5,
    n_removed_features: int = 0,
) -> List[List[Union[int, str, float]]]:
    """
    Calculate evaluation metrics by fold and append results to the given list.
    # noqa
    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        rs (bool): Random State.
        rpn (int): Random Permutation number.
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.
        ths_range (Union[float, List[float]]): Thresholds for binary classification.
        n_removed_features (int): Number of removed features.
        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """
    # If a float value was provided, convert in list for iteration
    if isinstance(ths_range, float):
        ths_range = [ths_range]

    for ths in ths_range:
        # Calculate the predictions using the passed ths
        prediction = (prob > ths).astype(int)

        # Compute all the metrics
        bacc = balanced_accuracy_score(y, prediction)
        auc = roc_auc_score(y, prob)
        f1 = f1_score(y, prediction)
        specificity = specificity_score(y, prediction)
        sensitivity = sensitivity_score(y, prediction)
        recall = recall_score(y, prediction)

        # Append results
        result.append(
            [
                i_fold,
                model,
                rs,
                rpn,
                ths,
                n_removed_features,
                bacc,
                auc,
                f1,
                specificity,
                sensitivity,
                recall,
            ]
        )

    return result


def results_to_df(result: List[List[Union[int, str, float]]]) -> pd.DataFrame:
    """
    Convert the list of results to a DataFrame.

    Parameters:
        result (List[List[Union[int, str, float]]]): List containing results.

    Returns:
        pd.DataFrame: DataFrame containing results with labeled columns.
    """
    result_df = pd.DataFrame(
        result,
        columns=[
            "Fold",
            "Model",
            "Random State",
            "Random Permutation Number",
            "Thresholds",
            "Number of Removed Features",
            "Balanced ACC",
            "AUC",
            "F1",
            "Specificity",
            "Sensitivity",
            "Recall",
        ],
    )
    return result_df
