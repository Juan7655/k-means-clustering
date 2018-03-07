import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd


class Kmeans:
	map_list = []
	c = None
	centroid_diff_list = []

	def __init__(self, k, data):
		self.k = k
		self.data = data

	def train(self, show_graph=False):
		now = datetime.datetime.now()
		if self.c is None:
			self.define_centroids()
		self.classify_points()

		self.define_centroids()
		self.classify_points()
		iterations = 0
		centroid_diff = 11

		while centroid_diff > 0.1:
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
			points_in_cluster.columns = ['i', 'x', 'y']
			points_in_cluster.drop(0)
			for j in range(len(points_in_cluster)):
				# act_point = points_in_cluster[points_in_cluster.columns[0], points_in_cluster.columns[1]][j]
				temp_p = np.array([points_in_cluster['x'][j], points_in_cluster['y'][j]])
				sum_distance += get_distance(temp_p, self.c[i])
			if show_graph:
				plt.scatter(points_in_cluster[points_in_cluster.columns[1]],  # x values
				            points_in_cluster[points_in_cluster.columns[2]],  # y values
				            zorder=1, s=5)
		if show_graph:
			plt.scatter(self.c.T[0], self.c.T[1], s=20, zorder=2, c='r')
			plt.show()
			plt.plot(self.centroid_diff_list)
			plt.show()
		return sum_distance

	def define_centroids(self):
		if self.c is None:
			self.c = np.random.rand(self.k, self.data.shape[1])
			return

		temp_map = np.array(self.map_list)
		for i in range(self.k):
			indices_in_cluster = temp_map[temp_map.T[1] == i].T[0]
			points_in_cluster = self.data.iloc[indices_in_cluster]
			means = points_in_cluster.mean(axis=0)
			self.c[i] = np.nan_to_num(means)

	def classify_points(self):
		self.map_list = []
		for i in range(len(self.data)):
			centroid, distance = 0, get_distance(self.data.iloc[[i]], self.c[0])
			for j in range(1, len(self.c)):
				temp_dist = get_distance(self.data.iloc[[i]], self.c[j])
				if temp_dist < distance:
					centroid, distance = j, temp_dist
			self.map_list.append([i, centroid])

	def centroid_diff(self, c):
		distance = 0
		for i in range(len(c)):
			distance += get_distance(c[i], self.c[i])
		return distance


def get_distance(p, q):
	diff = np.square(p - q).T
	try:
		diff.columns = [0]
		suma = np.sum(diff)
		result = suma[0] ** 0.5
	except AttributeError:
		result = np.sum(diff) ** 0.5
	except TypeError:
		print("lel")
	return result
