import pandas as pd
import joblib
from AnomalyDetectionFS import detect_with_pretrained

pretrained = joblib.load("pretrained_model.joblib")

# ----- Test with nominal scenario
# df_fault = pd.read_csv("./Data/Nominal/5.csv")

# ----- Fault: Initial temperature too high
# df_fault = pd.read_csv("./Data/Faults/fault_case_tr105.csv")
# df_fault = pd.read_csv("./Data/Faults/fault_case_tr110.csv")

# ----- Fault: Cooling degraded UA=5 after 10000s
# df_fault = pd.read_csv("./Data/Faults/fault_case_UA=5.csv")

# ----- Fault: Colling off after 10000s
# df_fault = pd.read_csv("./Data/Faults/test.csv") # fault_case_UA_off

# ----- Fault: Initial concentration of HP too high
df_fault = pd.read_csv("./Data/Faults/fault_case_HP0=10.15.csv")


df_res, det = detect_with_pretrained(
    df_fault, pretrained, persist_k=1, warmup_s=1500, use_gates=True
)

# print("Threshold =", det["threshold"])
print("Detected at t(s) =", det["t_detect_s"])

# df_res contient anomaly_score + anomaly_flag
# df_res.to_excel("fault_case_01_with_detection.xlsx", index=False)
