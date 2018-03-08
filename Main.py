import Kmeans
import pandas as pd
import matplotlib.pyplot as plt


def run():
	# data for multi-dimensionality (4 features)
	data = pd.read_csv("results4-feat.csv")
	# dataset with 2 features for testing graph and visualizations
	# data = pd.read_csv("results.csv")

	while True:
		plot_distances(data, max_val=5)
	# model = Kmeans.Kmeans(4, data)
	# model.train(show_graph=True)


def plot_distances(data, max_val, min_val=2):
	distances = []
	for i in range(min_val, max_val + 1):
		model = Kmeans.Kmeans(i, data)
		distances.append(model.train(show_graph=False))
	plt.plot([i + 2 for i in range(len(distances))], distances)
	plt.xlabel("Number of clusters")
	plt.ylabel("Total Sum")
	plt.title("Elbow Method")
	plt.show()


if __name__ == "__main__":
	run()
