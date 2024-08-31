from MachineLearningProcedure import MachineLearningProcedure


if __name__ == '__main__':

    MachineLearningProcedure(5, variant=("h1n1", "seas"), steps=("mi",), store=False, mi_models=("lr",), dp_short=False).main()
