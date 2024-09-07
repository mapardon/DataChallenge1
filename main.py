from multiprocessing import Process

from MachineLearningProcedure import MachineLearningProcedure
from SpecificIdentification import multi_proc

if __name__ == '__main__':

    # MachineLearningProcedure(5, variant=("h1n1", "seas"), steps=("mi",),  # pre, mi, exp store=True, mi_models=("ada", "gbc", "bc"), dp_short=False).main()

    procs = [Process(target=MachineLearningProcedure(5, ("h1n1", "seas"), ("pre",), False, ("ada", "gbc", "bc",), False).main),
             Process(target=MachineLearningProcedure(5, ("h1n1", "seas"), ("mi",), False, ("ada", "gbc", "bc",), False).main),
             Process(target=multi_proc)]

    for p in procs:
        p.start()

    for p in procs:
        p.join()
