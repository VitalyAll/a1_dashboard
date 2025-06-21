import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

st.set_page_config(layout="wide")

st.title("\U0001F4C8 Визуализация ускорений A1 по ρ и E")

# === Путь к папке с CSV-файлами на Google Drive ===
results_folder = 'data'

# === Загрузка данных ===
@st.cache_data
def load_data(folder):
    records = []
    for filename in os.listdir(folder):
        if filename.endswith('.csv') and filename.startswith('rho'):
            match = re.search(r'rho(\d+)_E(\d+)MPa', filename)
            if not match:
                continue
            rho = int(match.group(1))
            E = int(match.group(2))
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                records.append({
                    'rho': rho,
                    'E': E,
                    't': row['Time'],
                    'a': row[df.columns[1]]
                })
    return pd.DataFrame(records)

df_all = load_data(results_folder)

# === Уникальные значения времени ===
t_vals = np.sort(df_all['t'].unique())
t_idx = st.slider("Выберите момент времени (сек)", 0, len(t_vals)-1, 0, format="%d")
t_val = t_vals[t_idx]

# === Тепловая карта ===
st.subheader(f"Тепловая карта ускорения A1 на t = {t_val:.3f} с")
df_t = df_all[np.isclose(df_all['t'], t_val, atol=1e-6)]
pivot = df_t.pivot_table(index='rho', columns='E', values='a')
pivot = pivot.sort_index().sort_index(axis=1).fillna(0)

fig, ax = plt.subplots(figsize=(10, 6))
c = ax.imshow(
    pivot.values,
    extent=[pivot.columns.min()/1e6, pivot.columns.max()/1e6,
            pivot.index.min(), pivot.index.max()],
    origin='lower',
    aspect='auto',
    cmap='coolwarm',
    interpolation='bicubic'
)
plt.colorbar(c, ax=ax, label='A1')
X, Y = np.meshgrid(pivot.columns / 1e6, pivot.index)
ax.contour(X, Y, pivot.values, colors='k', linewidths=0.5, alpha=0.6)
ax.set_xlabel('E (МПа)')
ax.set_ylabel('ρ (кг/м³)')
ax.set_title(f'A1 при t = {t_val:.3f} с')
st.pyplot(fig)

# === Анализ минимальных ускорений ===
st.subheader("🔍 Минимальное среднее ускорение |A1| по модулю")
df_all['abs_a'] = df_all['a'].abs()
mean_abs = df_all.groupby(['rho', 'E'])['abs_a'].mean().reset_index()
min_mean = mean_abs['abs_a'].min()
min_points = mean_abs[mean_abs['abs_a'] == min_mean]
st.dataframe(min_points)

# === Тепловая карта средних ускорений ===
st.subheader("Среднее по модулю A1 по всей временной шкале")
pivot_mean = mean_abs.pivot(index='rho', columns='E', values='abs_a').sort_index().sort_index(axis=1)
fig2, ax2 = plt.subplots(figsize=(10, 6))
c2 = ax2.imshow(
    pivot_mean.values,
    extent=[pivot_mean.columns.min()/1e6, pivot_mean.columns.max()/1e6,
            pivot_mean.index.min(), pivot_mean.index.max()],
    origin='lower',
    aspect='auto',
    cmap='YlGnBu',
    interpolation='bicubic'
)
plt.colorbar(c2, ax=ax2, label='Mean |A1|')
X2, Y2 = np.meshgrid(pivot_mean.columns / 1e6, pivot_mean.index)
ax2.contour(X2, Y2, pivot_mean.values, colors='k', linewidths=0.5, alpha=0.6)
ax2.set_xlabel('E (МПа)')
ax2.set_ylabel('ρ (кг/м³)')
ax2.set_title('Средние ускорения по модулю')
st.pyplot(fig2)
