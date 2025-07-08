# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import RobustScaler

# # Load the dataset
# df = pd.read_csv("dados_consolidados_pontoB.csv")

# # Drop the date column
# df_process = df.drop(columns=["calendar_date"])

# # Separate target and features
# y_target = df_process['cvd_risk']
# x_features = df_process.drop(columns=['cvd_risk'])

# # Apply RobustScaler
# scaler = RobustScaler()
# x_scaled = pd.DataFrame(scaler.fit_transform(x_features), columns=x_features.columns)

# # Plotting the scaled data
# plt.figure(figsize=(14, 6))
# sns.boxplot(data=x_scaled)
# plt.xticks(rotation=90)
# plt.title("Boxplot of Scaled Features using RobustScaler")
# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import numpy as np

# Load and seperate the features and target columns
df = pd.read_csv("dados_consolidados_pontoB.csv")

df_process = df.drop(columns=["calendar_date"])
y_target = df_process['cvd_risk']
x_features = df_process.drop(columns=['cvd_risk'])

# Scale features
scaler = RobustScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x_features, y_target), columns=x_features.columns)

X_scaled_contiguous = np.ascontiguousarray(x_scaled.values, dtype=np.float64)

print(X_scaled_contiguous.flags)

# Reduce to 2D with PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

# Create a DataFrame for plotting
pca_df = pd.DataFrame(x_pca, columns=["PC1", "PC2"])
pca_df["cvd_risk"] = y_target.values

# Scatter plot with points for checking class
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cvd_risk", palette="viridis", s=60, alpha=0.8)
plt.title("Scatter Plot of Scaled Features (via PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# # Plotting the scaled data
plt.figure(figsize=(14, 6))
sns.boxplot(data=x_scaled)
plt.xticks(rotation=90)
plt.title("Scaled Features using RobustScaler")
plt.tight_layout()
plt.show()