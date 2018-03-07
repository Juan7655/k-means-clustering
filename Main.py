import Kmeans
import pandas as pd
from mpi4py import MPI
import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def run():
	data = pd.read_csv("results3d.csv")
	print("my rank is: " + str(rank))
	print("my size: " + str(comm.Get_size()))

	plot_distances(data, max_val=10)
	# model = Kmeans.Kmeans(4, data)
	# model.train(show_graph=True)


def plot_distances(data, max_val, min_val=2):
	distances = []
	for i in range(min_val, max_val):
		model = Kmeans.Kmeans(i, data)
		distances.append(model.train(show_graph=False))
	plt.plot([i + 2 for i in range(len(distances))], distances)
	plt.xlabel("Number of clusters")
	plt.ylabel("Total Sum")
	plt.show()


if __name__ == "__main__":
	run()
