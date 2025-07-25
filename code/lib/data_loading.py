import pandas as pd
from lib.data_processing import remove_low_variance_features


def load_icu_discharge_data(datadir, variance_ths):
    endpoint_dir = datadir
    cols_to_keep = ["Patch ID", "Geschlecht", "Alter", "Rehospitalisierung", "Tod"]

    endpoint = pd.read_excel(
        endpoint_dir / "Endpunkt_Mortalität_und_Rehospitalisierung(2).xlsx",
        header=1,
        usecols=cols_to_keep,
    )
    endpoint.drop(0, axis=0, inplace=True)

    # # Assign the highest NT-proBNP level to expired patients

    endpoint["Rehospitalisierung"].replace({"n.A. ": 0, "n. A. ": 0}, inplace=True)
    # Assign the highest NT-proBNP level to expired patients
    endpoint["Tod"].replace({"n. A. ": 0}, inplace=True)
    endpoint["Target"] = endpoint["Rehospitalisierung"].values + endpoint["Tod"].values
    endpoint.drop(columns=["Rehospitalisierung", "Tod"], inplace=True)
    endpoint.rename(columns={"Alter": "Age"}, inplace=True)
    endpoint.rename(columns={"Patch ID": "Patient_ID"}, inplace=True)
    endpoint.rename(columns={"Geschlecht": "Sex"}, inplace=True)
    endpoint["Patient_ID"] = endpoint["Patient_ID"].apply(lambda x: x.split("_")[1])

    use_cols = [
        "Patient_ID",
        "Heart_Rate_mean",
        "Heart_Rate_std",
        "Respiratory_Rate_mean",
        "Respiratory_Rate_std",
        "RtoR_Interval_mean",
        "Standart_Deviation_NN_interval",
    ]

    vp_dir = datadir

    # Read the CSV file and load only the specified columns
    vital_patch_features = pd.read_csv(
        vp_dir / "vital_patch_icu_discharge.csv", usecols=use_cols
    )

    # merge table and vital patch data.
    total_data = pd.merge(endpoint, vital_patch_features, on="Patient_ID", how="inner")

    Y = total_data["Target"].to_numpy()
    #
    total_data.drop(columns=["Patient_ID", "Target"], inplace=True)
    #
    # Remove low variance features
    X = remove_low_variance_features(total_data, variance_ths)
    X = X.astype(float, errors="ignore")

    return X, Y
