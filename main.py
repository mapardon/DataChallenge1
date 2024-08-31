from MachineLearningProcedure import MachineLearningProcedure


if __name__ == '__main__':

    MachineLearningProcedure(5, variant=("h1n1", "seas"), steps=("pre", "mi", "exp"), store=False, mi_models=("lm",), dp_short=True).main()
