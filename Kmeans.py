import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Kmeans:
	map_list = []
	c = None
	centroid_diff_list = []

	def __init__(self, k, data):
		self.k = k
		self.data = data
		self.features = data.shape[1]

	def train(self, show_graph=False):
		now = datetime.datetime.now()
		if show_graph and self.features > 2:
			print("No plotted graph: input data exceeds dimensionality")
			show_graph = False

		if self.c is None:
			self.define_centroids()
		self.classify_points()

		self.define_centroids()
		self.classify_points()
		iterations = 0
		centroid_diff = 11

		while centroid_diff != 0.0:
			iterations += 1
			actual_c = self.c.copy()
			self.classify_points()
			self.define_centroids()
			print(str(iterations) + ": " + str(self.c))
			centroid_diff = self.centroid_diff(actual_c)
			self.centroid_diff_list.append(centroid_diff)
		print("Iterations: " + str(iterations))
		print("Centroid difference: " + str(centroid_diff))
		print("Execution time: " + str(datetime.datetime.now() - now))
		temp_map = np.array(self.map_list)
		sum_distance = 0
		for i in range(self.k):
			indices_in_cluster = temp_map[temp_map.T[1] == i].T[0]
			points_in_cluster = pd.DataFrame(self.data.iloc[indices_in_cluster]).reset_index()
			points_in_cluster = pd.DataFrame(points_in_cluster.drop(['index'], axis=1))
			for j in range(len(points_in_cluster)):
				# act_point = points_in_cluster[points_in_cluster.columns[0], points_in_cluster.columns[1]][j]
				temp_p = np.array([points_in_cluster[k][j] for k in points_in_cluster.columns])
				sum_distance += get_distance(temp_p, self.c[i])
			if show_graph:
				plt.scatter(points_in_cluster[points_in_cluster.columns[0]],  # x values
				            points_in_cluster[points_in_cluster.columns[1]],  # y values
				            zorder=1, s=5)
		if show_graph:
			plt.scatter(self.c.T[0], self.c.T[1], s=20, zorder=2, c='r')
			plt.show()
			plt.plot(self.centroid_diff_list)
			plt.show()
		return sum_distance

	# for each cluster, defines its new centroid's coordinates, calculating mean-point of classified points
	def define_centroids(self):
		if self.c is None:
			self.c = np.random.rand(self.k, self.features)
			self.c = np.array(self.data.min(axis=0)) + self.c * np.array((self.data.max(axis=0) - self.data.min(axis=0)))
			return

		temp_map = np.array(self.map_list)
		for i in range(self.k):
			# gets the indices of the points classified in the i cluster
			indices_in_cluster = temp_map[temp_map.T[1] == i].T[0]
			# select the points classified in the i cluster
			points_in_cluster = self.data.iloc[indices_in_cluster]
			# calculates the mean value of the classified data points
			means = points_in_cluster.mean(axis=0)
			self.c[i] = np.nan_to_num(means)

	# Creates a list to relate each point's index with the actual classification
	def classify_points(self):
		self.map_list = []
		n = len(self.data)

		for i in range(n * rank / size, n * (rank + 1) / size):
			centroid, distance = 0, get_distance(self.data.iloc[[i]], self.c[0])
			for j in range(1, len(self.c)):
				temp_dist = get_distance(self.data.iloc[[i]], self.c[j])
				if temp_dist < distance:
					centroid, distance = j, temp_dist
			self.map_list.append([i, centroid])

	# Calculates the distance moved by the centroids from one iteration to another
	def centroid_diff(self, c):
		distance = 0
		for i in range(len(c)):
			distance += get_distance(c[i], self.c[i])
		return distance


def get_distance(p, q, n=8):
	diff = (np.abs(p - q)**n).T
	try:
		diff.columns = [0]
		suma = np.sum(diff)
		result = suma[0] ** (1/n)
	except AttributeError:
		result = np.sum(diff) ** (1/n)
	return result
