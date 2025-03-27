import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# load the dataset
iris_data = pd.read_csv(r"C:\Users\basil\OneDrive\Desktop\Base\Undergraduate\Semester 4\F78DS - Data Science Life Cycle\iris.csv")

# standardize column names
iris_data.columns = iris_data.columns

# scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(iris_data[['sepallength', 'sepalwidth']])

# scatter plot of sepal width vs sepal length
plt.scatter(iris_data['sepalwidth'], iris_data['sepallength'])
plt.xlabel('sepal width (cm)')
plt.ylabel('sepal length (cm)')
plt.title('sepal width vs sepal length')
plt.show()

# function to find optimal k using the elbow method
def optimise_k_means(data, max_k):
    means, inertias = [], []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        means.append(k)
        inertias.append(kmeans.inertia_)

    # elbow plot
    plt.figure(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('number of clusters')
    plt.ylabel('inertia')
    plt.title('elbow method for optimal k')
    plt.grid(True)
    plt.show()

# run elbow method on selected features
optimise_k_means(iris_data[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']], 10)

# applying k-means with k=2
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(scaled_features)

# add cluster labels to dataframe
iris_data['kmeans_2'] = kmeans.labels_

# scatter plot with cluster colors
plt.scatter(iris_data['sepalwidth'], iris_data['sepallength'], c=iris_data['kmeans_2'], cmap='viridis')
plt.xlabel('sepal width (cm)')
plt.ylabel('sepal length (cm)')
plt.title('k-means clustering (k=2)')
plt.colorbar(label='cluster')
plt.show()

# create multiple clusters (k=1 to 5) and plot each
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    iris_data[f'kmeans_{k}'] = kmeans.labels_

    # scatter plot for each k
    plt.scatter(iris_data['sepalwidth'], iris_data['sepallength'], c=iris_data[f'kmeans_{k}'], cmap='viridis')
    plt.xlabel('sepal width (cm)')
    plt.ylabel('sepal length (cm)')
    plt.title(f'k-means clustering (k={k})')
    plt.colorbar(label='cluster')
    plt.show()

# display the first few rows with cluster labels
print(iris_data.head())
