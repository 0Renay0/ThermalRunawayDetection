import pandas as pd
import joblib
from AnomalyDetectionFS import detect_with_pretrained

pretrained = joblib.load("pretrained_model.joblib")

df_fault = pd.read_csv("./Data/Faults/fault_case_01.csv")

df_res, det = detect_with_pretrained(df_fault, pretrained, persist_k=3)

print("Threshold =", det["threshold"])
print("Detected at t(s) =", det["t_detect_s"])

# df_res contient anomaly_score + anomaly_flag
# df_res.to_excel("fault_case_01_with_detection.xlsx", index=False)
