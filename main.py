import time
from multiprocessing import Process

from MachineLearningProcedure import MachineLearningProcedure


def multi_proc():
    n_exp_rounds = 4
    variants = ("h1n1", "seas")
    steps = ("pi",)  # steps: pre, pi, si, exp
    dp_model_tag = "lr"
    pi_pars1 = (("bc", "max_features_lr"), ("bc", "max_features_gbc"),)
    pi_pars2 = (("bc", "max_features_lr"), ("bc", "n_estimators_gbc"),)
    si_models = ("lm", "lr")
    procs = [Process(target=MachineLearningProcedure(exp_rounds=n_exp_rounds, variants=variants, steps=steps,
                                                     store=True, short_ds=False, dp_model_tag=dp_model_tag,
                                                     pi_pars=pi_pars1, si_models=si_models).main),
             Process(target=MachineLearningProcedure(exp_rounds=n_exp_rounds, variants=variants, steps=steps,
                                                     store=True, short_ds=False, dp_model_tag=dp_model_tag,
                                                     pi_pars=pi_pars2, si_models=si_models).main)]

    for p in procs: p.start()
    for p in procs: p.join()


def uni_proc():
    n_exp_rounds = 5
    variants = ("h1n1", "seas")
    steps = ("si", "exp")  # steps: pre, pi, si, exp
    short_ds = False
    dp_model_tag = "lr"
    pi_pars = (("gbc", "n_estimators"),)
    si_models = ("bc",)
    MachineLearningProcedure(n_exp_rounds, variants=variants, steps=steps, store=True, short_ds=short_ds,
                             dp_model_tag=dp_model_tag, pi_pars=pi_pars, si_models=si_models).main()


def test_uni():
    n_exp_rounds = 1
    variants = ("h1n1", "seas")
    steps = ("si",)  # steps: pre, pi, si, exp
    short_ds = True
    dp_model_tag = "lr"
    pi_pars = (("bc", "n_estimators_gbc"),)
    si_models = ("bc",)
    MachineLearningProcedure(exp_rounds=n_exp_rounds, variants=variants, steps=steps, store=False, short_ds=short_ds,
                             dp_model_tag=dp_model_tag, pi_pars=pi_pars, si_models=si_models).main()


def test_multi():
    n_exp_rounds = 1
    variants = ("h1n1", "seas")
    steps = ("pi",)  # steps: pre, pi, si, exp
    short_ds = True
    dp_model_tag = "lr"
    pi_pars1 = (("bc", "max_features_lr"), ("bc", "max_features_gbc"),)
    pi_pars2 = (("bc", "max_features_lr"), ("bc", "n_estimators_gbc"),)
    si_models = ("lm", "lr")
    procs = [Process(target=MachineLearningProcedure(exp_rounds=n_exp_rounds, variants=variants, steps=steps,
                                                     store=False, short_ds=short_ds, dp_model_tag=dp_model_tag,
                                                     pi_pars=pi_pars1, si_models=si_models).main),
             Process(target=MachineLearningProcedure(exp_rounds=n_exp_rounds, variants=variants, steps=steps,
                                                     store=False, short_ds=short_ds, dp_model_tag=dp_model_tag,
                                                     pi_pars=pi_pars2, si_models=si_models).main)]

    for p in procs: p.start()
    for p in procs: p.join()


if __name__ == '__main__':

    t = time.time()
    uni_proc()
    print("runtime:", time.time() - t)
