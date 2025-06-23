import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
data = pd.read_csv("C:/Users/madhu/Downloads/archive (2)/Mall_Customers.csv")
df = pd.DataFrame(data)
x = df[['Annual Income (k$)','Spending Score (1-100)']]
scaler = scaler = StandardScaler()
scaled_features = scaler.fit_transform(x)
k=3
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(x)
plt.figure(figsize=(8, 6))
for cluster in range(k):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {cluster}') 
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments by K-Means Clustering')
plt.legend()
plt.show()
