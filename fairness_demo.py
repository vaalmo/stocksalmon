import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Datos sintéticos reproducibles
SEED = 42
rs = np.random.RandomState(SEED)
N = 4000

# Atributo sensible A (dos grupos)
A = rs.binomial(1, 0.5, size=N)

# Una característica con shift y cambio de varianza por grupo
x = rs.normal(loc=0.0 + 0.8*A, scale=1.0 + 0.2*A, size=N)

# Score "verdadero" (probabilidad) y etiqueta binaria
logit = 0.6 * x + (-0.4) + 0.6 * A
p = 1 / (1 + np.exp(-logit))        # será nuestro "score" de modelo (sin dependencias externas)
y = rs.binomial(1, p)

df = pd.DataFrame({"x": x, "A": A, "y": y, "score": p})


# Helpers de métricas
def confusion_from_threshold(y_true, scores, thr):
    """Devuelve TP, FP, TN, FN usando scores >= thr como predicción positiva."""
    y_pred = (scores >= thr).astype(int)
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return TP, FP, TN, FN, y_pred

def metrics_from_confusion(TP, FP, TN, FN):
    P = TP + FN
    Nn = TN + FP
    sel = (TP + FP) / (P + Nn + 1e-12)              # tasa de selección (DP)
    tpr = TP / (P + 1e-12)                          # recall (EO)
    fpr = FP / (Nn + 1e-12)
    ppv = TP / (TP + FP + 1e-12)                    # precisión positiva (PPV)
    acc = (TP + TN) / (P + Nn + 1e-12)
    return dict(selection_rate=sel, TPR=tpr, FPR=fpr, PPV=ppv, ACC=acc, P=P, N=Nn)

def group_metrics_from_thresholds(df, t0, t1):
    """Calcula métricas por grupo aplicando t0 para A=0 y t1 para A=1."""
    mask0 = (df["A"] == 0); mask1 = ~mask0
    TP0, FP0, TN0, FN0, yhat0 = confusion_from_threshold(df.loc[mask0,"y"].values,
                                                         df.loc[mask0,"score"].values, t0)
    TP1, FP1, TN1, FN1, yhat1 = confusion_from_threshold(df.loc[mask1,"y"].values,
                                                         df.loc[mask1,"score"].values, t1)
    m0 = metrics_from_confusion(TP0, FP0, TN0, FN0); m0.update({"TP":TP0,"FP":FP0,"FN":FN0})
    m1 = metrics_from_confusion(TP1, FP1, TN1, FN1); m1.update({"TP":TP1,"FP":FP1,"FN":FN1})
    return {0: m0, 1: m1}, yhat0, yhat1

def dp_gap(res): 
    return abs(res[1]["selection_rate"] - res[0]["selection_rate"])

def eo_gap(res): 
    return abs(res[1]["TPR"] - res[0]["TPR"])

def make_row(name, t0, t1, res):
    return {
        "Setup": name,
        "Threshold G0": float(t0),
        "Threshold G1": float(t1),
        "DP gap": dp_gap(res),
        "EO gap": eo_gap(res),
        "ACC G0": res[0]["ACC"],
        "ACC G1": res[1]["ACC"],
        "SelRate G0": res[0]["selection_rate"],
        "SelRate G1": res[1]["selection_rate"],
        "TPR G0": res[0]["TPR"],
        "TPR G1": res[1]["TPR"],
        "PPV G0": res[0]["PPV"],
        "PPV G1": res[1]["PPV"],
    }

def pretty_block(title, thresholds, res):
    print(f"\n{title}")
    if title.startswith("BASE MODEL"):
        print(f"Brecha Paridad Demográfica (DP) = {dp_gap(res)}")
        print(f"Brecha Igualdad de Oportunidad (EO) = {eo_gap(res)}")
    else:
        print(f"Umbrales: Grupo0 = {thresholds[0]} Grupo1 = {thresholds[1]}")
        print(f"Brecha DP = {dp_gap(res)}")
        print(f"Brecha EO = {eo_gap(res)}")
    print({
        0: {k: res[0][k] for k in ["TP","FP","FN","P","N","selection_rate","TPR","FPR","PPV","ACC"]},
        1: {k: res[1][k] for k in ["TP","FP","FN","P","N","selection_rate","TPR","FPR","PPV","ACC"]}
    })


# Baseline (umbral global 0.5)
t_base = 0.5
res_base, yhat0_b, yhat1_b = group_metrics_from_thresholds(df, t_base, t_base)


# EO dinámico: igualar TPR
def find_threshold_for_target_tpr(y_true, scores, target_tpr):
    """
    Busca el umbral cuyo TPR esté más cerca del target_tpr.
    Explora todos los scores únicos como candidatos de umbral (más bordes 0 y 1).
    """
    candidates = np.unique(np.concatenate([[0.0, 1.0], scores]))
    best_thr, best_gap = 0.5, 1e9
    for thr in candidates:
        TP, FP, TN, FN, _ = confusion_from_threshold(y_true, scores, thr)
        P = TP + FN
        tpr = TP / (P + 1e-12)
        gap = abs(tpr - target_tpr)
        if gap < best_gap:
            best_gap = gap
            best_thr = thr
    return float(best_thr)

# TPR de baseline por grupo y target como promedio
mask0 = (df["A"]==0); mask1 = ~mask0
TP0, FP0, TN0, FN0, _ = confusion_from_threshold(df.loc[mask0,"y"].values,
                                                 df.loc[mask0,"score"].values, t_base)
TP1, FP1, TN1, FN1, _ = confusion_from_threshold(df.loc[mask1,"y"].values,
                                                 df.loc[mask1,"score"].values, t_base)
tpr0_base = TP0 / (TP0 + FN0 + 1e-12)
tpr1_base = TP1 / (TP1 + FN1 + 1e-12)
tpr_target = 0.5*(tpr0_base + tpr1_base)

# Umbrales que logran TPR cercano al target para cada grupo
t0_eo = find_threshold_for_target_tpr(df.loc[mask0,"y"].values, df.loc[mask0,"score"].values, tpr_target)
t1_eo = find_threshold_for_target_tpr(df.loc[mask1,"y"].values, df.loc[mask1,"score"].values, tpr_target)
res_eo, yhat0_eo, yhat1_eo = group_metrics_from_thresholds(df, t0_eo, t1_eo)


# DP dinámico: igualar selección sin saturar
# Objetivo de selección: el promedio de las selecciones baseline (para no empujar a 100%)
sel0_base = (df.loc[mask0,"score"] >= t_base).mean()
sel1_base = (df.loc[mask1,"score"] >= t_base).mean()
target_sel = 0.5*(sel0_base + sel1_base)

# Búsqueda en rejilla (rápida) para hallar (t0, t1)
cands = np.linspace(0.05, 0.95, 181)
best = None
best_loss = 1e9

for t0 in cands:
    # fijamos t0 -> sel0 y luego buscamos t1 que minimice diferencia y cercanía al target
    sel0 = (df.loc[mask0,"score"] >= t0).mean()
    # para acelerar, evaluamos t1 en cands y escogemos el mejor
    losses = []
    for t1 in cands:
        sel1 = (df.loc[mask1,"score"] >= t1).mean()
        # pérdida = brecha DP + penalización por alejarse del target medio
        loss = abs(sel0 - sel1) + 0.5*abs(0.5*(sel0 + sel1) - target_sel)
        losses.append((loss, t1, sel0, sel1))
    loss, t1_opt, sel0_tmp, sel1_tmp = min(losses, key=lambda z: z[0])
    if loss < best_loss:
        best_loss = loss
        best = (t0, t1_opt)

t0_dp, t1_dp = best
res_dp, yhat0_dp, yhat1_dp = group_metrics_from_thresholds(df, t0_dp, t1_dp)


#Tabla resumen + CSV 
rows = [
    make_row("Base (0.50)", t_base, t_base, res_base),
    make_row(f"EO (t0={t0_eo:.3f}, t1={t1_eo:.3f})", t0_eo, t1_eo, res_eo),
    make_row(f"DP (t0={t0_dp:.3f}, t1={t1_dp:.3f})", t0_dp, t1_dp, res_dp),
]
tab = pd.DataFrame(rows)[[
    "Setup","Threshold G0","Threshold G1","DP gap","EO gap",
    "ACC G0","ACC G1","SelRate G0","SelRate G1","TPR G0","TPR G1","PPV G0","PPV G1"
]]

print("\n=== TABLA RESUMEN ===")
print(tab.to_string(index=False))

os.makedirs("figs", exist_ok=True)
tab.to_csv("fairness_results.csv", index=False)


pretty_block("BASE MODEL (thresh=0.5):", (t_base, t_base), res_base)
pretty_block("MITIGACIÓN EO (umbrales distintos por grupo):", (t0_eo, t1_eo), res_eo)
pretty_block("MITIGACIÓN DP (umbrales distintos por grupo):", (t0_dp, t1_dp), res_dp)


# Gráficas simples 

# a) Brechas DP y EO
plt.figure()
xpos = np.arange(len(tab))
plt.bar(xpos - 0.2, tab["DP gap"].values, width=0.4, label="DP gap")
plt.bar(xpos + 0.2, tab["EO gap"].values, width=0.4, label="EO gap")
plt.xticks(xpos, ["Base","EO","DP"])
plt.ylabel("Brecha (abs)")
plt.title("Brechas de Fairness por configuración")
plt.legend()
plt.tight_layout()
plt.savefig("figs/fairness_gaps.png", dpi=150)
plt.close()

# b) Accuracy promedio
plt.figure()
acc_avg = 0.5*(tab["ACC G0"].values + tab["ACC G1"].values)
plt.bar(np.arange(len(acc_avg)), acc_avg)
plt.xticks(np.arange(len(acc_avg)), ["Base","EO","DP"])
plt.ylabel("Accuracy promedio")
plt.title("Accuracy promedio por configuración")
plt.tight_layout()
plt.savefig("figs/accuracy_avg.png", dpi=150)
plt.close()

# c) Tasa de selección por grupo
plt.figure()
idx = np.arange(len(tab))
plt.bar(idx - 0.2, tab["SelRate G0"].values, width=0.4, label="G0")
plt.bar(idx + 0.2, tab["SelRate G1"].values, width=0.4, label="G1")
plt.xticks(idx, ["Base","EO","DP"])
plt.ylabel("Tasa de selección")
plt.title("Tasa de selección por grupo")
plt.legend()
plt.tight_layout()
plt.savefig("figs/selection_rates.png", dpi=150)
plt.close()

print("\nGuardado:")
print("- fairness_results.csv")
print("- figs/fairness_gaps.png")
print("- figs/accuracy_avg.png")
print("- figs/selection_rates.png")
