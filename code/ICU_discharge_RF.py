# %%
import numpy as np
import pandas as pd
from lib.data_loading import load_icu_discharge_data
from lib.ml_utils import compute_results, results_to_df
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
project_dir = Path().resolve().parents[0]
datadir = project_dir / "data"
save_dir = project_dir / "output" / "RF"

# %%
variance_ths = 0.10
X, Y = load_icu_discharge_data(datadir, variance_ths=variance_ths)
# Minimun feature variance
# Set random state
random_state = 23

# Cross validation parameters
out_n_splits = 3
out_n_repetitions = 10


n_estimators = 3

# number of thresholds used
ths_range = list(np.linspace(0, 1, 101))


# Final data shape
n_participants, n_features = X.shape
# Show the feature distribution
print("Features: " + str(n_features))
print("Participants: " + str(n_participants))

# %%
kf_out = RepeatedStratifiedKFold(
    n_splits=out_n_splits, n_repeats=out_n_repetitions, random_state=random_state
)

# Initialize variables
results = []

predictions = []
y_true_loop = []
imp_mean = IterativeImputer(random_state=random_state)

score_clf = RandomForestClassifier(n_estimators=n_estimators)
# Outer loop
for i_fold, (train_index, test_index) in enumerate(kf_out.split(X=X, y=Y)):
    print("FOLD: " + str(i_fold))

    # Patients used for train and internal XGB validation
    X_train_whole = X.iloc[train_index, :]
    Y_train_whole = Y[train_index]

    # Patients used to generete a prediction
    X_test = X.iloc[test_index, :]
    Y_test = Y[test_index]

    # impute train data, round for matching with the original distribution
    X_train_whole_imputed = np.round(imp_mean.fit_transform(X_train_whole))
    # impute test data, round for matching with the original distribution
    X_test_imputed = np.round(imp_mean.transform(X_test))

    print("Fitting RF model")
    score_clf.fit(X=X_train_whole_imputed, y=Y_train_whole)

    # Get probability
    pred_test = score_clf.predict_proba(X_test_imputed)[:, 1]

    pred_train = score_clf.predict_proba(X_train_whole_imputed)[:, 1]

    # Compute metrics
    predictions.append(pred_test)
    y_true_loop.append(Y_test)  # noqa

    # Compute metrics without removing any feature
    results = compute_results(
        i_fold,
        f"ICU dischargeresults_to_df (RF_{n_estimators}",
        pred_test,
        Y_test,
        results,
        ths_range=ths_range,
    )
    results = compute_results(
        i_fold,
        f"ICU discharge (RF_{n_estimators}) Train",
        pred_train,
        Y_train_whole,
        results,
        ths_range=ths_range,
    )

# Create a dataframe to save
results_df = results_to_df(results)

# %%

# % Saving results
print("Saving Results")
results_df.to_csv(save_dir / f"icu_discharge_VP_model_RF_{n_estimators}.csv")  # noqa

predictions_full = pd.DataFrame(predictions)
predictions_full = predictions_full.T
predictions_full.to_csv(save_dir / f"icu_discharge_predictions_VP_model_RF_{n_estimators}.csv")  # noqa

y_true_loop = pd.DataFrame(y_true_loop)
y_true_loop = y_true_loop.T
y_true_loop.to_csv(save_dir / f"icu_discharge_y_true_VP_model_RF_{n_estimators}.csv")  # noqa


# %%
