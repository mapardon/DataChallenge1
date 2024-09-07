from multiprocessing import Process

from MachineLearningProcedure import MachineLearningProcedure


if __name__ == '__main__':

    # MachineLearningProcedure(5, variant=("h1n1", "seas"), steps=("mi",),  # pre, mi, exp store=True, mi_models=("ada", "gbc", "bc"), dp_short=False).main()

    procs = list()
    for ml_conf in [(5, ("h1n1", "seas"), ("pre",), False, ("lm",), True),
                    (5, ("h1n1", "seas"), ("mi",), False, ("lm",), True)]:
        procs.append(Process(target=MachineLearningProcedure(*ml_conf).main))
        procs[-1].start()

    for p in procs:
        p.join()
