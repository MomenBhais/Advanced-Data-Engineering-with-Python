# Data Engineering Guide: NumPy, Pandas, and Visualization

This guide covers essential concepts and practical implementations of **NumPy**, **Pandas**, and **Data Visualization** using **Matplotlib** and **Seaborn**—core libraries for numerical computing, data manipulation, and visualization in **Python**.

## 📌 Table of Contents
- NumPy Overview
- Pandas Overview
- Data Visualization Overview
- Advanced Data Processing
- Performance Optimization
- Data Preprocessing
- Feature Engineering

---

# 📊 NumPy Overview

## 🚀 Getting Started
```bash
pip install numpy
```
```python
import numpy as np
```

## 🔹 NumPy Arrays
```python
arr = np.array([1, 2, 3])
print(arr.shape, arr.dtype)
```

## 🔹 Mathematical Operations
```python
arr * 2, np.dot(arr, arr)
```

## 🔹 Indexing and Slicing
```python
print(arr[1:])
```

---

# 🐼 Pandas Overview

## 🚀 Getting Started
```bash
pip install pandas
```
```python
import pandas as pd
```

## 🔹 Creating and Manipulating DataFrames
```python
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
print(df.head())
```

## 🔹 Data Cleaning
```python
df.fillna(0, inplace=True)
```

## 🔹 Exporting Data
```python
df.to_csv('file.csv', index=False)
```

---

# 📊 Data Visualization Overview

## 🚀 Getting Started
```bash
pip install matplotlib seaborn
```
```python
import matplotlib.pyplot as plt
import seaborn as sns
```

## 🔹 Matplotlib Basics
```python
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

## 🔹 Seaborn for Advanced Visualization
```python
sns.histplot(df['A'])
plt.show()
```

---

# 🚀 Advanced Data Processing

## 🔹 Handling Missing Data
```python
df.dropna(inplace=True)  # Remove missing values
df.fillna(df.mean(), inplace=True)  # Fill missing values with mean
```

## 🔹 Data Transformation
```python
df['C'] = df['A'] * 2  # Create a new column
```

## 🔹 Merging and Joining DataFrames
```python
df2 = pd.DataFrame({'A': [1, 2], 'D': [5, 6]})
merged_df = pd.merge(df, df2, on='A', how='inner')
```

---

# ⚡ Performance Optimization

## 🔹 Using Vectorized Operations
```python
df['A'] = df['A'] * 2  # Faster than iterating over rows
```

## 🔹 Efficient Data Types
```python
df['A'] = df['A'].astype('int16')  # Reduce memory usage
```

## 🔹 Using Multi-threading with Dask
```bash
pip install dask
```
```python
import dask.dataframe as dd
ddf = dd.from_pandas(df, npartitions=2)
```

---

# 🛠️ Data Preprocessing

## 🔹 Handling Categorical Data
```python
df['Category'] = df['Category'].astype('category')
```

## 🔹 Scaling and Normalization
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['A', 'B']] = scaler.fit_transform(df[['A', 'B']])
```

## 🔹 Encoding Categorical Variables
```python
pd.get_dummies(df, columns=['Category'])
```

---

# 🎯 Feature Engineering

## 🔹 Creating New Features
```python
df['A_B_Sum'] = df['A'] + df['B']
```

## 🔹 Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(df[['A', 'B']], df['Target'])
```

## 🔹 Dimensionality Reduction
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[['A', 'B']])
```

---

✅ **Stay tuned for more advanced topics in Data Engineering! 🚀**

---

✍️ **Prepared by:** Momen Mohammed Bhais  
📩 **Contact:** [momenbhais@outlook.com](mailto:momenbhais@outlook.com)
