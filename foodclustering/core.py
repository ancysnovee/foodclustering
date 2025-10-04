import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class FoodCluster:
    def __init__(self, data_folder=None, n_clusters=5):
        """
        Initialize the FoodCluster class:
        - Loads CSVs from the data folder
        - Merges data
        - Performs scaling and KMeans clustering
        """
        # Automatically locate data folder if not given
        if data_folder is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_folder = os.path.join(current_dir, "data")
        else:
            data_folder = os.path.abspath(data_folder)

        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        # List all CSV files in the folder
        file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".csv")]
        if not file_paths:
            raise FileNotFoundError(f"No CSV files found in: {data_folder}")

        # Load and merge datasets
        dfs = [pd.read_csv(path) for path in file_paths]
        self.df = pd.concat(dfs, ignore_index=True)

        # Rename columns for consistency
        self.df.rename(columns={"food": "food_name"}, inplace=True)

        # Fill missing features
        self.features = ["Caloric Value", "Fat", "Carbohydrates", "Sugars", "Protein", "Sodium"]
        for feat in self.features:
            if feat not in self.df.columns:
                self.df[feat] = 0
        self.df[self.features] = self.df[self.features].fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.features])

        # KMeans clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df["Cluster"] = self.kmeans.fit_predict(X_scaled)

    def check_food(self, food_name: str):
        """
        Returns the food info including cluster and health status
        """
        food_row = self.df[self.df["food_name"].str.lower() == food_name.lower()]
        if food_row.empty:
            return f"Food '{food_name}' not found in dataset."

        row = food_row.iloc[0]
        cluster = int(row["Cluster"])
        calories = float(row["Caloric Value"])
        protein = float(row["Protein"])
        fat = float(row["Fat"])
        sugars = float(row.get("Sugars", 0))

        # Improved health logic
        if calories <= 150 and fat <= 5 and sugars <= 15:
            status = "Healthy"
        elif calories <= 300 and fat <= 15 and sugars <= 30:
            status = "Moderate"
        else:
            status = "Unhealthy"

        return {
            "food": row["food_name"].title(),
            "cluster": cluster,
            "calories": calories,
            "protein": protein,
            "fat": fat,
            "sugars": sugars,
            "status": status
        }

    def recommend_similar(self, food_name: str, top_n=5):
        """
        Returns top N similar foods from the same cluster
        """
        food_row = self.df[self.df["food_name"].str.lower() == food_name.lower()]
        if food_row.empty:
            return f"Food '{food_name}' not found in dataset."

        cluster_id = int(food_row["Cluster"].values[0])
        cluster_foods = self.df[self.df["Cluster"] == cluster_id]["food_name"].tolist()

        # Remove the queried food itself
        cluster_foods = [f for f in cluster_foods if f.lower() != food_name.lower()]
        return cluster_foods[:top_n]

    def plot_clusters(self, feature_x="Caloric Value", feature_y="Protein"):
        """
        Scatter plot of clusters using any two features
        """
        if feature_x not in self.df.columns or feature_y not in self.df.columns:
            print("Invalid feature names.")
            return

        plt.figure(figsize=(10, 6))
        for cluster in sorted(self.df["Cluster"].unique()):
            cluster_data = self.df[self.df["Cluster"] == cluster]
            plt.scatter(cluster_data[feature_x], cluster_data[feature_y], label=f"Cluster {cluster}", alpha=0.6)

        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(f"Food Clusters ({feature_x} vs {feature_y})")
        plt.legend()
        plt.show()


# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    cluster_model = FoodCluster(n_clusters=5)

    # Check food info
    print(cluster_model.check_food("cola coca cola"))

    # Get similar foods
    print(cluster_model.recommend_similar("cola coca cola"))

    # Visualize clusters
    cluster_model.plot_clusters(feature_x="Caloric Value", feature_y="Protein")
