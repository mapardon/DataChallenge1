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
    MachineLearningProcedure(2, variants=("h1n1",), steps=("pre",), store=False,
                             dp_model="lr", mi_models=("lr",), dp_short=True).main()


if __name__ == '__main__':

    uni_proc()
