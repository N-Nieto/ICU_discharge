import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def remove_low_variance_features(
    data: pd.DataFrame, variance_ths: float
) -> pd.DataFrame:
    """
    Remove low-variance features from a DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - variance_ths (float): The variance threshold for feature selection.

    Returns:
    - pd.DataFrame: The DataFrame with low-variance features removed.
    """

    # Record the number of features before the removal
    feature_pre = data.shape[1]

    # Create a VarianceThreshold instance with the specified threshold
    selector = VarianceThreshold(threshold=variance_ths)

    # Fit the selector to your data
    selector.fit(data)

    # Get the indices of the selected features
    selected_features_indices = selector.get_support(indices=True)

    # Filter the DataFrame to keep only the selected features
    data = data.iloc[:, selected_features_indices]

    # Record the number of features after the removal
    feature_post = data.shape[1]

    # Print the number of features deleted
    print(str(feature_pre - feature_post) + " Features deleted")

    return data
