import pulp
import numpy as np
import pandas as pd
from collections import defaultdict

# ========================= PARAMÈTRES =========================
cap_limit_frac = 0.95   # plafond d’utilisation des EGC
rho_tilt = 0.1           # biais dans phase C2
eps_disp = 1e-3           # pondération dans phase C1

# ============================ DONNÉES =========================
ops = [f"O{str(i).zfill(2)}" for i in range(1, 21)]
egcs = ["E1 (3)", "E2 (2)", "E3 (4)", "E4 (3)", "E5 (2)",
        "E6 (5)", "E7 (2)", "E8 (3)", "E9 (3)", "E10 (2)"]

need_matrix = np.array([
    [0.10, 0.15, 0.16, 0.90, 0.70, 0.60,   np.nan, np.nan, np.nan, np.nan],
    [0.12, 1.99, 0.34, 0.01, 1.66, 0.13,   np.nan, np.nan, np.nan, np.nan],
    [1.57, 0.19, 4.42, 2.00, 0.04, 2.11,   np.nan, np.nan, np.nan, np.nan],
    [0.11, 0.13, 0.08, 0.07, 0.11, 2.29,   np.nan, np.nan, np.nan, np.nan],
    [0.48, 1.96, 0.02, 0.13, 0.43, 3.72,   np.nan, np.nan, np.nan, np.nan],
    [1.02, 4.27, 1.80,   np.nan, 3.87, 0.67,   np.nan, np.nan, np.nan, np.nan],
    [1.40,   np.nan, 2.63, 1.69, 0.30, 0.94,   np.nan, np.nan, np.nan, np.nan],
    [1.94, 2.64, 1.36,   np.nan, 1.11, 2.37,   np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, 4.41, 1.89, np.nan, 2.21,   np.nan, np.nan, np.nan, np.nan],
    [3.01, np.nan, 3.94, 3.13, 2.62, 3.78,   np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan, 1.85, np.nan,   np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan, 1.79, np.nan,   np.nan, 3.32, np.nan, np.nan],
    [2.75, 3.26, np.nan, np.nan, np.nan, 3.82,   np.nan, np.nan, 3.25, np.nan],
    [0.13, np.nan, 0.69, np.nan, 3.50, np.nan,   np.nan, np.nan, 3.45, 0.78],
    [0.49, 2.24, np.nan, np.nan, np.nan, np.nan,   np.nan, 3.07, 4.13, np.nan],
    [np.nan, np.nan, np.nan, 3.59, np.nan, np.nan, 1.27, np.nan, 0.04, 1.95],
    [2.81, np.nan, 2.64, np.nan, 3.04, 3.79,   np.nan, np.nan, np.nan, 0.37],
    [3.26, np.nan, 0.07, np.nan, np.nan, 2.82,   np.nan, np.nan, 2.67, np.nan],
    [np.nan, np.nan, np.nan, np.nan, 1.75, np.nan,   np.nan, np.nan, np.nan, np.nan],
    [1.60, np.nan, 3.17, 3.43, np.nan, np.nan,   np.nan, np.nan, 4.48, np.nan]
])

needs_df = pd.DataFrame(need_matrix, index=ops, columns=egcs)
capacities = {egc: int(egc.split("(")[1].split(")")[0]) for egc in egcs}
cap_lim = {j: cap_limit_frac * capacities[j] for j in egcs}

needs = {}
for i in ops:
    for j in egcs:
        val = needs_df.loc[i, j]
        if not pd.isna(val) and val > 0:
            needs[(i, j)] = val

ops_allowed = defaultdict(list)
for (i, j) in needs:
    ops_allowed[i].append(j)

mono_ops = [i for i, lst in ops_allowed.items() if len(lst) == 1]
mono_load_j = {j: 0 for j in egcs}
for i in mono_ops:
    j = ops_allowed[i][0]
    mono_load_j[j] += needs[(i, j)]
y_req = {j: max(0, mono_load_j[j] - cap_lim[j]) for j in egcs}

def print_utilisation(t_vals, phase_name):
    print(f"--- Taux d’utilisation après {phase_name} ---")
    for j, tv in t_vals.items():
        print(f"{j}: {round(100 * tv, 2)}%")
    print()

# ========================= PHASE F0 =========================
probF0 = pulp.LpProblem("PhaseF0", pulp.LpMinimize)
x = pulp.LpVariable.dicts("x", needs.keys(), lowBound=0, upBound=1)
y_extra = pulp.LpVariable.dicts("y_extra", egcs, lowBound=0)

probF0 += pulp.lpSum(y_extra[j] for j in egcs)
for i in ops:
    probF0 += pulp.lpSum(x[(i, j)] for j in ops_allowed[i]) == 1
for j in egcs:
    probF0 += pulp.lpSum(
        needs[(i2, j)] * x[(i2, j)]
        for (i2, j2) in needs if j2 == j
    ) <= cap_lim[j] + y_req[j] + y_extra[j]

probF0.solve(pulp.PULP_CBC_CMD(msg=False))
xF0 = {k: x[k].value() for k in needs}
y_extra_F0 = {j: y_extra[j].value() for j in egcs}
cap_eff = {j: cap_lim[j] + y_req[j] + y_extra_F0[j] for j in egcs}

# calcul des t (utilisation) après F0
tF0 = {}
for j in egcs:
    total_need_allocated = sum(needs[(i2, j)] * xF0[(i2, j)]
                               for (i2, j2) in needs if j2 == j)
    # t_j = (charge affectée) / capacity
    tF0[j] = total_need_allocated / capacities[j]
print_utilisation(tF0, "Phase F0")

# ========================= PHASE A =========================
probA = pulp.LpProblem("PhaseA", pulp.LpMaximize)
xA = pulp.LpVariable.dicts("xA", needs.keys(), lowBound=0, upBound=1)
t = pulp.LpVariable.dicts("t", egcs, lowBound=0)
m = pulp.LpVariable("m", lowBound=0)

probA += m
for i in ops:
    probA += pulp.lpSum(xA[(i, j)] for j in ops_allowed[i]) == 1
for j in egcs:
    probA += pulp.lpSum(
        needs[(i2, j)] * xA[(i2, j)]
        for (i2, j2) in needs if j2 == j
    ) == capacities[j] * t[j]
    probA += pulp.lpSum(
        needs[(i2, j)] * xA[(i2, j)]
        for (i2, j2) in needs if j2 == j
    ) <= cap_eff[j]
    probA += t[j] >= m

probA.solve(pulp.PULP_CBC_CMD(msg=False))
xA_val = {k: xA[k].value() for k in needs}
tA = {j: t[j].value() for j in egcs}
m_star = m.value()
print_utilisation(tA, "Phase A")

# ========================= PHASE B =========================
probB = pulp.LpProblem("PhaseB", pulp.LpMinimize)
xB = pulp.LpVariable.dicts("xB", needs.keys(), lowBound=0, upBound=1)
tB = pulp.LpVariable.dicts("tB", egcs, lowBound=0)
U = pulp.LpVariable("U", lowBound=0)

probB += U
for i in ops:
    probB += pulp.lpSum(xB[(i, j)] for j in ops_allowed[i]) == 1
for j in egcs:
    probB += pulp.lpSum(
        needs[(i2, j)] * xB[(i2, j)]
        for (i2, j2) in needs if j2 == j
    ) == capacities[j] * tB[j]
    probB += pulp.lpSum(
        needs[(i2, j)] * xB[(i2, j)]
        for (i2, j2) in needs if j2 == j
    ) <= cap_eff[j]
    probB += tB[j] >= m_star
    if mono_load_j.get(j, 0) == 0:
        probB += tB[j] <= U

probB.solve(pulp.PULP_CBC_CMD(msg=False))
xB_val = {k: xB[k].value() for k in needs}
tB_vals = {j: tB[j].value() for j in egcs}
U_star = U.value()
print_utilisation(tB_vals, "Phase B")

# ========================= PHASE C1 =========================
probC1 = pulp.LpProblem("PhaseC1", pulp.LpMinimize)
xC1 = pulp.LpVariable.dicts("xC1", needs.keys(), lowBound=0, upBound=1)
tC1 = pulp.LpVariable.dicts("tC1", egcs, lowBound=0)
d = pulp.LpVariable.dicts("d", egcs, lowBound=0)
D = pulp.LpVariable("D", lowBound=0)

probC1 += D + eps_disp * pulp.lpSum(d[j] for j in egcs)
for i in ops:
    probC1 += pulp.lpSum(xC1[(i, j)] for j in ops_allowed[i]) == 1
for j in egcs:
    probC1 += pulp.lpSum(
        needs[(i2, j)] * xC1[(i2, j)]
        for (i2, j2) in needs if j2 == j
    ) == capacities[j] * tC1[j]
    probC1 += pulp.lpSum(
        needs[(i2, j)] * xC1[(i2, j)]
        for (i2, j2) in needs if j2 == j
    ) <= cap_eff[j]
    probC1 += tC1[j] >= m_star
    if mono_load_j.get(j, 0) == 0:
        probC1 += tC1[j] <= U_star
    probC1 += tC1[j] - m_star <= d[j]
    probC1 += m_star - tC1[j] <= d[j]

share_pairs = []
for j in egcs:
    for k in egcs:
        if j < k and any((i, j) in needs and (i, k) in needs for i in ops):
            share_pairs.append((j, k))
for (j, k) in share_pairs:
    probC1 += tC1[j] - tC1[k] <= D
    probC1 += tC1[k] - tC1[j] <= D

probC1.solve(pulp.PULP_CBC_CMD(msg=False))
xC1_val = {k: xC1[k].value() for k in needs}
tC1_vals = {j: tC1[j].value() for j in egcs}
D_star = D.value()
print_utilisation(tC1_vals, "Phase C1")

# ========================= PHASE C2 =========================
probC2 = pulp.LpProblem("PhaseC2", pulp.LpMinimize)
xC2 = pulp.LpVariable.dicts("xC2", needs.keys(), lowBound=0, upBound=1)
tC2 = pulp.LpVariable.dicts("tC2", egcs, lowBound=0)
d2 = pulp.LpVariable.dicts("d2", egcs, lowBound=0)

obj = pulp.lpSum(d2[j] for j in egcs)
for (i, j) in needs:
    obj += rho_tilt * (tA[j] - m_star) * xC2[(i, j)]
probC2 += obj

for i in ops:
    probC2 += pulp.lpSum(xC2[(i, j)] for j in ops_allowed[i]) == 1
for j in egcs:
    probC2 += pulp.lpSum(
        needs[(i2, j)] * xC2[(i2, j)]
        for (i2, j2) in needs if j2 == j
    ) == capacities[j] * tC2[j]
    probC2 += pulp.lpSum(
        needs[(i2, j)] * xC2[(i2, j)]
        for (i2, j2) in needs if j2 == j
    ) <= cap_eff[j]
    probC2 += tC2[j] >= m_star
    if mono_load_j.get(j, 0) == 0:
        probC2 += tC2[j] <= U_star
    probC2 += tC2[j] - m_star <= d2[j]
    probC2 += m_star - tC2[j] <= d2[j]

for (j, k) in share_pairs:
    probC2 += tC2[j] - tC2[k] <= D_star
    probC2 += tC2[k] - tC2[j] <= D_star

probC2.solve(pulp.PULP_CBC_CMD(msg=False))
xC2_val = {k: xC2[k].value() for k in needs}
tC2_vals = {j: tC2[j].value() for j in egcs}
print_utilisation(tC2_vals, "Phase C2 (final)")

# Affichage final de la répartition
repartition = pd.DataFrame(index=ops, columns=egcs).fillna(0.0)
for (i, j), val in xC2_val.items():
    repartition.loc[i, j] = round(100 * val, 2)

print("\n Répartition finale (en %) :")
print(repartition)
