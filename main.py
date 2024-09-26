import time
from multiprocessing import Process

from MachineLearningProcedure import MachineLearningProcedure


def multi_proc():

    gbc_pars = [("gbc", "n_estimators"), ("gbc", "subsample"), ("gbc", "min_sample_split"), ("gbc", "max_depth")]
    nn_pars = [("nn", "hl1"), ("nn", "hl2"), ("nn", "act_f"), ("n", "solver"), ("nn", "miter")]
    procs = [Process(target=MachineLearningProcedure(5, ("h1n1", "seas"), ("pi",), True, "lm", gbc_pars, None, False).main),
             Process(target=MachineLearningProcedure(5, ("h1n1", "seas"), ("pi",), True, "lm", nn_pars, None, False).main)]

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def uni_proc():
    # steps: pre, pi, si, exp
    mi_pars = (("gbc", "n_estimators"),)
    MachineLearningProcedure(3, variants=("h1n1", "seas"), steps=("pi",), store=True,
                             dp_model_tag="lr", pi_pars=mi_pars, si_models=("lm", "lr", "nn",), short_ds=False).main()


def test():
    mi_pars = (("gbc", "n_estimators"),)
    MachineLearningProcedure(3, variants=("h1n1",), steps=("pre",), store=False,
                             dp_model_tag="lr", pi_pars=mi_pars, si_models=("lm", "lr", "nn",), short_ds=True).main()


if __name__ == '__main__':

    t = time.time()
    test()
    print("runtime:", time.time() - t)
