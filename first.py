from foodclustering import FoodCluster

cluster_model = FoodCluster(n_clusters=5)

print(cluster_model.check_food("banana"))
print(cluster_model.recommend_similar("banana"))
cluster_model.plot_clusters()

