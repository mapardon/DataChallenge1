import statistics

from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification


if __name__ == '__main__':

    for m1 in ["knn", "remove", "mean", "median", "most_frequent"]:
        for m2 in ["remove", "most_frequent"]:
            if [m1, m2].count("remove") == 1:
                continue
            tmp = list()
            for _ in range(5):
                tmp.append(ModelIdentification(*DataPreprocessing().data_preprocessing(m1, m2), 5).main())
            print("{}/{}: avg: {}, stdev: {}".format(m1, m2, statistics.mean(tmp), statistics.stdev(tmp)))
