import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

class FoodCluster:
    def __init__(self):
        # Automatically find all CSVs in the 'data' folder
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, "data")
        csv_files = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) if f.endswith(".csv")
        ]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the 'data' folder!")

        # Load and merge datasets
        dfs = [pd.read_csv(f).loc[:, ~pd.read_csv(f).columns.str.contains("^Unnamed")] for f in csv_files]
        self.df = pd.concat(dfs, ignore_index=True)

        # Rename for consistency
        self.df.rename(columns={"food": "food_name"}, inplace=True)

        # Features for clustering
        self.features = ["Caloric Value", "Fat", "Carbohydrates", "Sugars", "Protein", "Sodium"]
        self.df[self.features] = self.df[self.features].fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.features])

        # KMeans clustering
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.df["Cluster"] = self.kmeans.fit_predict(X_scaled)

    def check_food(self, food_name: str):
        food_row = self.df[self.df["food_name"].str.lower() == food_name.lower()]
        if food_row.empty:
            return {"error": f"Food '{food_name}' not found."}

        row = food_row.iloc[0]
        calories = row["Caloric Value"]
        protein = row["Protein"]
        fat = row["Fat"]
        cluster = int(row["Cluster"])

        # Simple healthy/unhealthy classification
        status = "Healthy" if calories <= 200 and fat <= 10 and row["Sugars"] <= 15 else "Unhealthy"

        return {
            "food": row["food_name"].title(),
            "cluster": cluster,
            "calories": float(calories),
            "protein": float(protein),
            "fat": float(fat),
            "status": status
        }

    def recommend_similar(self, food_name: str, top_n=5):
        food_row = self.df[self.df["food_name"].str.lower() == food_name.lower()]
        if food_row.empty:
            return {"error": f"Food '{food_name}' not found."}

        cluster_id = int(food_row["Cluster"].values[0])
        cluster_foods = self.df[self.df["Cluster"] == cluster_id]["food_name"].tolist()

        # Exclude the searched food itself
        cluster_foods = [f for f in cluster_foods if f.lower() != food_name.lower()]

        return cluster_foods[:top_n]

# ------------------------------
# TESTING WHEN RUN DIRECTLY
# ------------------------------
if __name__ == "__main__":
    cluster_model = FoodCluster()

    test_foods = ["banana", "cheddar cheese", "eggnog", "pizza"]
    for food in test_foods:
        info = cluster_model.check_food(food)
        similar = cluster_model.recommend_similar(food)
        print(info)
        print("Similar foods:", similar)
        print("-"*50)
