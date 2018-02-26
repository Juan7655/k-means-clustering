import numpy as np


class Kmeans:
	map_list = []
	c = None

	def __init__(self, k, data):
		self.k = k
		self.data = data

	def train(self):
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
			print(self.c)
			centroid_diff = self.centroid_diff(actual_c)
		print(iterations)
		print(centroid_diff)

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
	return result
