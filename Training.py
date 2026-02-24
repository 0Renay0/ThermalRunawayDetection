import glob
import pandas as pd
import joblib

from AnomalyDetectionFS import fit_nominal_model

# Load nominal datasets
paths = sorted(glob.glob("./Data/Nominal/*.csv"))
nominal_datasets = [pd.read_csv(path) for path in paths]


# Train the model on nominal datasets
pretrained = fit_nominal_model(
    nominal_datasets=nominal_datasets,
    time_col="Time",
    T_col="Tr_K",
    P_col="Pression_ideal_bar",
    win=15,
    contamination=0.001,
    thr_quantile=0.99,
    random_state=0,
)

print("Threshold nominal =", pretrained["threshold"])

# Save the pretrained model
joblib.dump(pretrained, "pretrained_model.joblib")
print("Pretrained model saved to pretrained_model.joblib")

# Test
"""
df_test = nominal_datasets[0]
df_out, det = detect_with_pretrained(df_test, pretrained, persist_k=3)
print("t_detect_s (should be None or very late) =", det["t_detect_s"])
"""
