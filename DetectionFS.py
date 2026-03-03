import pandas as pd
import joblib
from AnomalyDetectionFS import detect_with_pretrained

pretrained = joblib.load("pretrained_model.joblib")

# ----- Test with nominal scenario
df_N1 = pd.read_csv("./Data/Nominal/1.csv")
df_N2 = pd.read_csv("./Data/Nominal/2.csv")
df_N3 = pd.read_csv("./Data/Nominal/3.csv")
df_N4 = pd.read_csv("./Data/Nominal/4.csv")
df_N5 = pd.read_csv("./Data/Nominal/5.csv")

# ----- Fault: Initial temperature too high
df_fault1 = pd.read_csv("./Data/Faults/fault_case_tr105.csv")
df_fault2 = pd.read_csv("./Data/Faults/fault_case_tr110.csv")

# ----- Fault: Cooling degraded UA=8 after 10000s
df_fault3 = pd.read_csv("./Data/Faults/fault_case_UA=8.csv")

# ----- Fault: Cooling degraded UA=5 after 10000s
df_fault4 = pd.read_csv("./Data/Faults/fault_case_UA=5.csv")

# ----- Fault: Colling off after 10000s
df_faul5 = pd.read_csv("./Data/Faults/fault_case_UA_off.csv")  #

# ----- Fault: Initial concentration of HP too high
df_faul6 = pd.read_csv("./Data/Faults/fault_case_HP0=10.15.csv")


print("Testing on nominal dataset")

df_res, det = detect_with_pretrained(
    df_N1, pretrained, persist_k=1, warmup_s=500, use_gates=True
)

print("N1- Detected at t(s) =", det["t_detect_s"])

df_res, det = detect_with_pretrained(
    df_N2, pretrained, persist_k=1, warmup_s=500, use_gates=True
)

print("N2- Detected at t(s) =", det["t_detect_s"])


df_res, det = detect_with_pretrained(
    df_N3, pretrained, persist_k=1, warmup_s=500, use_gates=True
)

print("N3- Detected at t(s) =", det["t_detect_s"])

df_res, det = detect_with_pretrained(
    df_N4, pretrained, persist_k=1, warmup_s=500, use_gates=True
)

print("N4- Detected at t(s) =", det["t_detect_s"])

df_res, det = detect_with_pretrained(
    df_N5, pretrained, persist_k=1, warmup_s=500, use_gates=True
)

print("N5- Detected at t(s) =", det["t_detect_s"])


print("Testing on faulty datasets")
df_res, det = detect_with_pretrained(
    df_fault1, pretrained, persist_k=1, warmup_s=500, use_gates=True
)
print("Fault1 (Tr0=105C) - Detected at t(s) =", det["t_detect_s"])

df_res, det = detect_with_pretrained(
    df_fault2, pretrained, persist_k=1, warmup_s=500, use_gates=True
)
print("Fault2 (Tr0=110C) - Detected at t(s) =", det["t_detect_s"])

df_res, det = detect_with_pretrained(
    df_fault3, pretrained, persist_k=1, warmup_s=500, use_gates=True
)
print("Fault3 (UA=8) - Detected at t(s) =", det["t_detect_s"])

df_res, det = detect_with_pretrained(
    df_fault4, pretrained, persist_k=1, warmup_s=500, use_gates=True
)
print("Fault4 (UA=5) - Detected at t(s) =", det["t_detect_s"])

df_res, det = detect_with_pretrained(
    df_faul5, pretrained, persist_k=1, warmup_s=500, use_gates=True
)
print("Fault5 (UA off) - Detected at t(s) =", det["t_detect_s"])

df_res, det = detect_with_pretrained(
    df_faul6, pretrained, persist_k=1, warmup_s=500, use_gates=True
)
print("Fault6 (HP0=10.15) - Detected at t(s) =", det["t_detect_s"])
