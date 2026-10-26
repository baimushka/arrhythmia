import pandas as pd
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv("data/raw/arrhythmia.data", header=None, na_values="?")
target_col = df.columns[-1]

# 2. Сбор статистики
stats = {
    "total_rows": df.shape[0],
    "total_cols": df.shape[1],
    "missing_values_total": int(df.isna().sum().sum()),
    "class_01_count": int(df[target_col].value_counts().get(1, 0)),
    "class_16_count": int(df[target_col].value_counts().get(16, 0)),
    "numeric_cols": df.select_dtypes(include=['number']).shape[1],
    "object_cols": df.select_dtypes(exclude=['number']).shape[1]
}
pd.DataFrame([stats]).to_csv("reports/LAB1/eda_stats.csv", index=False)
print("1. CSV со статистикой сохранен!")

# 3. Построение графика
# Группируем и считаем количество каждого класса
class_counts = df[target_col].value_counts().sort_index()

plt.figure(figsize=(12, 6))
# Строим столбчатую диаграмму
class_counts.plot(kind="bar", color="skyblue", edgecolor="black")

plt.title("Распределение классов в датасете Arrhythmia")
plt.xlabel("Класс (1 = Норма)")
plt.ylabel("Количество примеров")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.savefig("reports/LAB1/class_imbalance.png", bbox_inches="tight")
plt.close()
plt.figure(figsize=(10, 4))
sns.boxplot(x=df[0], color='lightgreen')
plt.title('Boxplot для признака 0 (Возраст)')
plt.xlabel('Значение')
plt.savefig('reports/LAB1/outliers_boxplot.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
corr_matrix = df[[0, 1, 2, 3, target_col]].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Матрица корреляций (первые 3 признака + класс)')
plt.savefig('reports/LAB1/correlation_matrix.png', bbox_inches='tight')
plt.close()
