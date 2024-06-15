from DataPreprocessing import DataPreprocessing
from ModelIdentification import ModelIdentification


if __name__ == '__main__':

    ModelIdentification(*DataPreprocessing().data_preprocessing(), 5).main()
