import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def get_status(calories, fat, sugar=0):
    """
    Determine health status of a food.
    Healthy: low calories, low fat, low sugar
    Moderate: medium
    Unhealthy: high
    """
    if calories <= 150 and fat <= 3 and sugar <= 15:
        return "Healthy"
    elif calories <= 300 and fat <= 10 and sugar <= 30:
        return "Moderate"
    else:
        return "Unhealthy"


class FoodCluster:
    def __init__(self, file_paths):
        # Load and merge all datasets
        dfs = [pd.read_csv(path) for path in file_paths]
        self.df = pd.concat(dfs, ignore_index=True)

        # Rename columns for consistency
        self.df.rename(columns={"food": "food_name"}, inplace=True)

        # Features to use for clustering
        self.features = ["Caloric Value", "Fat", "Carbohydrates", "Sugars", "Protein", "Sodium"]

        # Handle missing values
        self.df[self.features] = self.df[self.features].fillna(0)

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.features])

        # Fit clustering model
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.df["Cluster"] = self.kmeans.fit_predict(X_scaled)

        # Add health status
        self.df["Status"] = self.df.apply(
            lambda row: get_status(row["Caloric Value"], row["Fat"], row.get("Sugars", 0)),
            axis=1
        )

    def check_food(self, food_name: str):
        food_row = self.df[self.df["food_name"].str.lower() == food_name.lower()]
        if food_row.empty:
            return f"Food '{food_name}' not found in dataset."

        row = food_row.iloc[0]
        return {
            "food": row['food_name'].title(),
            "cluster": int(row["Cluster"]),
            "calories": row["Caloric Value"],
            "protein": row["Protein"],
            "fat": row["Fat"],
            "sugar": row.get("Sugars", 0),
            "status": row["Status"]
        }

    def recommend_similar(self, food_name: str, top_n=5):
        food_row = self.df[self.df["food_name"].str.lower() == food_name.lower()]
        if food_row.empty:
            return f"Food '{food_name}' not found in dataset."

        cluster_id = int(food_row["Cluster"].values[0])
        cluster_foods = self.df[self.df["Cluster"] == cluster_id].copy()

        # Remove the searched food itself
        cluster_foods = cluster_foods[cluster_foods["food_name"].str.lower() != food_name.lower()]

        # Prepare list of dicts with health status
        similar_list = []
        for _, row in cluster_foods.head(top_n).iterrows():
            similar_list.append({
                "food": row["food_name"].title(),
                "cluster": int(row["Cluster"]),
                "calories": row["Caloric Value"],
                "protein": row["Protein"],
                "fat": row["Fat"],
                "sugar": row.get("Sugars", 0),
                "status": row["Status"]
            })

        return similar_list

    def plot_clusters(self, x_feature="Calories", y_feature="Protein"):
        """
        Create scatter plot of clusters using two features.
        """
        if x_feature not in self.df.columns or y_feature not in self.df.columns:
            raise ValueError(f"Columns {x_feature} and {y_feature} must exist in dataset.")

        plt.figure(figsize=(10, 6))
        for cluster_id in sorted(self.df["Cluster"].unique()):
            cluster_data = self.df[self.df["Cluster"] == cluster_id]
            plt.scatter(
                cluster_data[x_feature],
                cluster_data[y_feature],
                label=f"Cluster {cluster_id}",
                alpha=0.6
            )
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(f"Food Clusters ({x_feature} vs {y_feature})")
        plt.legend()
        plt.grid(True)
        plt.show()


# ------------------------------
# TESTING WHEN RUN DIRECTLY
# ------------------------------
if __name__ == "__main__":
    # Paths to your datasets
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    file_paths = [
        os.path.join(data_folder, f"FOOD-DATA-GROUP{i}.csv") for i in range(1, 6)
    ]

    cluster_model = FoodCluster(file_paths)

    # Test examples
    foods_to_test = ["banana", "cheddar cheese", "eggnog"]
    for food in foods_to_test:
        print(cluster_model.check_food(food))
        print(cluster_model.recommend_similar(food))
        print("-" * 50)

    # Scatter plot example
    cluster_model.plot_clusters(x_feature="Caloric Value", y_feature="Protein")
