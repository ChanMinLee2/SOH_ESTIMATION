import pickle, pandas as pd, numpy as np

pkl_path = r"c:\Users\nisl_server2\Desktop\LFP_SOH_prediction\data_unified\HUST\1-8.pkl"
with open(pkl_path, "rb") as f:
    raw = pickle.load(f)
df = raw["cycles"]

# The gap is 36923 seconds ~ 10.26 hours - between discharge rows at positions 247 and 248
# Nothing exists globally in that gap
# This means: there is a real-time gap in the raw data (e.g. rest/pause period)
# But the cycle label assignment kept it as the same cycle 168, same phase "discharge"
# The trapezoidal integration across the gap multiplies V*I * 36923 seconds = huge energy

# Let"s verify: what does V*I*dt give for just the gap row?
# Row before gap: time=562789, V=2.9923, I=-2.19959
# Row after gap:  time=599712, V=3.1529, I=-2.19959
# Trapezoid energy for just that one step:
v1, v2 = 2.9923, 3.1529
i1, i2 = 2.19959, 2.19959
dt_gap = 599712 - 562789  # 36923 s
avg_vi = ((v1*i1) + (v2*i2)) / 2
energy_gap_Wh = avg_vi * dt_gap / 3600
print(f"Energy contribution from single gap step: {energy_gap_Wh:.3f} Wh")
print(f"dt_gap = {dt_gap} s = {dt_gap/3600:.2f} h")
print(f"This alone explains most of the 74.8 Wh anomaly")

# Check the normal cycle for context
dis_normal = df[df["cycle"] == 10][df["phase"] == "discharge"].sort_values("time_s")
t_n = dis_normal["time_s"].values
v_n = dis_normal["voltage_V"].values
i_n = np.abs(dis_normal["current_A"].values)
e_n = np.trapezoid(v_n * i_n, t_n) / 3600
print(f"\nCycle 10 discharge energy_Wh: {e_n:.3f}")
print(f"Cycle 10 t_discharge: {t_n.max() - t_n.min():.0f} s")

# Check if there are other HUST cells with similar gaps (let"s check cell 1-1)
pkl_1_1 = r"c:\Users\nisl_server2\Desktop\LFP_SOH_prediction\data_unified\HUST\1-1.pkl"
with open(pkl_1_1, "rb") as f:
    raw_1_1 = pickle.load(f)
df_1_1 = raw_1_1["cycles"]
# Check if any discharge in 1-1 has large dt
all_dis_1_1 = df_1_1[df_1_1["phase"] == "discharge"].sort_values(["cycle", "time_s"])
big_dts = []
for cyc_num, grp in all_dis_1_1.groupby("cycle"):
    grp_sorted = grp.sort_values("time_s")
    dts = np.diff(grp_sorted["time_s"].values)
    if len(dts) > 0 and dts.max() > 100:
        big_dts.append((cyc_num, dts.max()))
print(f"\nCell 1-1: cycles with discharge dt > 100s: {big_dts[:10]}")
print(f"Total such cycles in 1-1: {len(big_dts)}")
