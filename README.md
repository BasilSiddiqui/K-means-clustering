# K-Means Clustering on the Iris Dataset

## Overview
This project demonstrates K-Means clustering on the **Iris dataset**, focusing on clustering the data points based on **sepal length and width**. The goal is to visualize how the data groups into clusters and determine the optimal number of clusters using the **Elbow Method**.

## Dataset
The dataset used is the well-known **Iris dataset**, which contains **150 observations** with four features:
- `sepallength` (cm)
- `sepalwidth` (cm)
- `petallength` (cm)
- `petalwidth` (cm)
- `class` (species label - not used in clustering)

## Implementation Details
### **1. Data Preprocessing**
- The dataset is loaded using `pandas`.
- Column names are **converted to lowercase** for consistency.
- Features used for clustering (`sepallength`, `sepalwidth`) are **standardized** using `StandardScaler` to ensure all features contribute equally.

### **2. Scatter Plot of Sepal Dimensions**
A scatter plot is generated to visualize the natural spread of the data before clustering.

### **3. Finding the Optimal Number of Clusters (Elbow Method)**
- A function `optimise_k_means()` runs K-Means for values of **k = 1 to 10**.
- The **inertia (sum of squared distances)** is plotted against the number of clusters to determine the **optimal k** (Elbow Method).

### **4. Applying K-Means Clustering**
- K-Means is applied with **k=2** (as an example), and the cluster labels are stored in `iris_data['kmeans_2']`.
- A scatter plot visualizes the **two clusters**.

### **5. Generating Clusters for k=1 to 5**
- A loop iterates over values of **k from 1 to 5**, applying K-Means each time.
- The cluster labels are stored in dynamically named columns (e.g., `kmeans_1`, `kmeans_2`, etc.).
- A scatter plot is generated for **each k** to observe how clusters form at different values.

## Requirements
The following Python libraries are required:
```bash
pip install pandas matplotlib scikit-learn
```

## Running the Code
Simply run the Python script, and it will:
1. Load and preprocess the dataset.
2. Display an **initial scatter plot**.
3. Compute and display the **Elbow Method plot**.
4. Apply K-Means clustering for **k=2** and plot the results.
5. Loop through **k=1 to 5**, applying K-Means and visualizing each clustering result.

## Expected Output
- **Scatter plot of Sepal Width vs Sepal Length** (before clustering)
- **Elbow Method Plot** (for choosing k)
- **Scatter plots of clustered data** for k=2, 3, 4, and 5

## Author
Basil Rehan

---
This project provides a simple but effective demonstration of K-Means clustering and data visualization. 🚀
