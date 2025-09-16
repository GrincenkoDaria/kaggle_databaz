
# uber_project.py
# — jednoduchá EDA + predikcia MILES (3. ročník SŠ)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# -------------------------
# 1) Načítanie a rýchly prehľad
# -------------------------
CSV_PATH = "UberDataset.csv"  # daj do rovnakého priečinka ako je tento .py

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Nenašiel som {CSV_PATH}. Ulož CSV vedľa skriptu.")

df = pd.read_csv(CSV_PATH)

print("\n--- Hlava dát ---")
print(df.head())

print("\n--- Info ---")
print(df.info())

print("\n--- Popisné štatistiky ---")
print(df.describe(include="all"))

# Niektoré datasety majú riadok "Totals" v START_DATE -> vyhodíme
if "START_DATE" in df.columns:
    df = df[df["START_DATE"] != "Totals"].copy()

# -------------------------
# 2) Čistenie a základná EDA
# -------------------------
# Konverzia dátumu
if "START_DATE" in df.columns:
    df["START_DATE"] = pd.to_datetime(df["START_DATE"], errors="coerce")

# Chýbajúce PURPOSE spočítame (len info)
if "PURPOSE" in df.columns:
    missing_purpose = df["PURPOSE"].isna().sum()
    print(f"\nPočet chýbajúcich PURPOSE: {missing_purpose}")

# Vyhodíme riadky bez MILES alebo START/STOP/CATEGORY (na jednoduchý model)
need_cols = ["MILES", "START", "STOP", "CATEGORY", "PURPOSE", "START_DATE"]
for c in need_cols:
    if c not in df.columns:
        print(f"VAROVANIE: Stĺpec {c} v datasete chýba. Niektoré časti sa nemusia vykonať.")

df = df.dropna(subset=["MILES"]).copy()

# EDA grafy (uložíme aj do súborov)
os.makedirs("figures", exist_ok=True)

# 2.1 rozdelenie kategórií (ak existuje)
if "CATEGORY" in df.columns:
    plt.figure(figsize=(7,5))
    df["CATEGORY"].value_counts().plot(kind="bar")
    plt.title("Počet záznamov podľa CATEGORY")
    plt.xlabel("CATEGORY"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig("figures/01_category_bar.png"); plt.show()

# 2.2 histogram míľ
plt.figure(figsize=(7,5))
plt.hist(df["MILES"], bins=20, edgecolor="black")
plt.title("Rozdelenie MILES")
plt.xlabel("MILES"); plt.ylabel("Frekvencia")
plt.tight_layout(); plt.savefig("figures/02_miles_hist.png"); plt.show()

# 2.3 koláč PURPOSE (ak existuje a má dáta)
if "PURPOSE" in df.columns and df["PURPOSE"].notna().any():
    plt.figure(figsize=(10,7))
    purpose_counts = df["PURPOSE"].value_counts()
    purpose_counts.plot(kind="pie", autopct="%1.1f%%")
    plt.ylabel(""); plt.title("Podiel PURPOSE")
    plt.tight_layout(); plt.savefig("figures/03_purpose_pie.png"); plt.show()

# 2.4 míle v čase (ak máme dátum)
if "START_DATE" in df.columns and df["START_DATE"].notna().any():
    tmp = df.dropna(subset=["START_DATE"]).set_index("START_DATE").resample("D")["MILES"].sum()
    plt.figure(figsize=(10,4))
    plt.plot(tmp.index, tmp.values)
    plt.title("Súčet míľ po dňoch")
    plt.xlabel("Dátum"); plt.ylabel("Súčet MILES")
    plt.tight_layout(); plt.savefig("figures/04_miles_over_time.png"); plt.show()

# -------------------------
# 3) Tvorba jednoduchých rysov (features)
# -------------------------
# Deň v týždni a hodina zo START_DATE (ak máme dátum)
if "START_DATE" in df.columns and df["START_DATE"].notna().any():
    d = df["START_DATE"].dropna()
    df.loc[d.index, "DOW"] = d.dt.dayofweek  # 0=pondelok ... 6=nedeľa
    df.loc[d.index, "HOUR"] = d.dt.hour
else:
    # ak nemáme START_DATE, pridáme default 0
    df["DOW"] = 0
    df["HOUR"] = 0

# -------------------------
# 4) Príprava dát pre model
# -------------------------
target = "MILES"
cat_cols = []
for c in ["START", "STOP", "CATEGORY", "PURPOSE"]:
    if c in df.columns:
        cat_cols.append(c)

num_cols = ["DOW", "HOUR"]  # jednoduché číselné vstupy
use_cols = cat_cols + num_cols

# Drop riadky s chýbajúcimi vstupmi, nech je to jednoduché
X = df.dropna(subset=use_cols).copy()
y = X[target].values
X = X[use_cols].copy()

# One-hot na kategórie
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# -------------------------
# 5) Train / Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# -------------------------
# 6) Model: RandomForest (jednoduchý)
# -------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------------------------
# 7) Vyhodnotenie (a baseline)
# -------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\n=== Výsledky modelu ===")
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"R² : {r2:.3f}")

# Baseline: predpovedať priemer z tréningu
baseline_value = np.mean(y_train)
y_base = np.full_like(y_test, baseline_value)
mae_b = mean_absolute_error(y_test, y_base)
mse_b = mean_squared_error(y_test, y_base)
r2_b  = r2_score(y_test, y_base)

print("\n=== Baseline (priemer tréningu) ===")
print(f"MAE: {mae_b:.3f}")
print(f"MSE: {mse_b:.3f}")
print(f"R² : {r2_b:.3f}")

# Porovnávací graf
plt.figure(figsize=(10,4))
plt.plot(y_test[:200], label="Skutočné", alpha=.8)
plt.plot(y_pred[:200], label="Model", alpha=.8)
plt.plot(y_base[:200], label="Baseline", alpha=.8)
plt.title("Porovnanie (prvých ~200 bodov)")
plt.legend(); plt.tight_layout(); plt.savefig("figures/05_compare_model_baseline.png"); plt.show()

print("\nHotovo. Grafy nájdeš v priečinku 'figures/'.")
