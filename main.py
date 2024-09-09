from multiprocessing import Process

from MachineLearningProcedure import MachineLearningProcedure
from SpecificIdentification import multi_proc


def multi_proc():
    procs = [Process(target=MachineLearningProcedure(5, ("h1n1", "seas"), ("pre",), False, ("ada", "gbc", "bc",), False).main),
             Process(target=MachineLearningProcedure(5, ("h1n1", "seas"), ("mi",), False, ("ada", "gbc", "bc",), False).main),
             Process(target=multi_proc)]

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def uni_proc():
    # steps: pre, mi, exp
    MachineLearningProcedure(5, variant=("h1n1",), steps=("pre",), store=False,
                             mi_models=("lm",), dp_short=False).main()


if __name__ == '__main__':

    uni_proc()
