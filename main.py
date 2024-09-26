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
    n_exp_rounds = 2
    variants = ("h1n1", "seas")
    steps = ("pre",)  # steps: pre, pi, si, exp
    dp_model_tag = "lr"
    pi_pars = (("gbc", "n_estimators"),)
    si_models = ("lm", "lr")
    MachineLearningProcedure(n_exp_rounds, variants=variants, steps=steps, store=True, short_ds=False,
                             dp_model_tag=dp_model_tag, pi_pars=pi_pars, si_models=si_models).main()


def test():
    n_exp_rounds = 2
    variants = ("h1n1", "seas")
    steps = ("pre",)  # steps: pre, pi, si, exp
    short_ds = True
    dp_model_tag = "lr"
    pi_pars = (("gbc", "n_estimators"),)
    si_models = ("lm", "lr")
    MachineLearningProcedure(n_exp_rounds, variants=variants, steps=steps, store=False, short_ds=short_ds,
                             dp_model_tag=dp_model_tag, pi_pars=pi_pars, si_models=si_models).main()


if __name__ == '__main__':

    t = time.time()
    test()
    print("runtime:", time.time() - t)
