import Kmeans
import pandas as pd


def run():
	data = pd.read_csv("results.csv")
	model = Kmeans.Kmeans(2, data)
	model.train()
	print(model.c)


if __name__ == "__main__":
	run()
