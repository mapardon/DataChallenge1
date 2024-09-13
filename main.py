from multiprocessing import Process

from MachineLearningProcedure import MachineLearningProcedure


def multi_proc():
    procs = [Process(target=MachineLearningProcedure(5, ("h1n1", "seas"), ("pre",), False, "lm", ("ada", "gbc", "bc",), False).main),
             Process(target=MachineLearningProcedure(5, ("h1n1", "seas"), ("mi",), False, "lm", ("ada", "gbc", "bc",), False).main),
             Process(target=multi_proc)]

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def uni_proc():
    # steps: pre, mi, exp
    MachineLearningProcedure(3, variants=("h1n1", "seas"), steps=("exp",), store=True,
                             dp_model="lr", mi_models=("lm",), dp_short=False).main()


if __name__ == '__main__':

    uni_proc()
