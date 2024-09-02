from MachineLearningProcedure import MachineLearningProcedure


if __name__ == '__main__':

    MachineLearningProcedure(5,
                             variant=("h1n1", "seas"),
                             steps=("mi",),  # pre, mi, exp
                             store=True,
                             mi_models=("tree",),
                             dp_short=False).main()
