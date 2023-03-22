from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import itertools
import os

def mult_proc_apply(func, elements, initializer=None, disable_tqdm=False, nb_workers=12):
    nb_workers = min(cpu_count(), nb_workers)
    p = Pool(nb_workers, initializer=initializer)
    try:
        res = list(tqdm(p.imap(func, elements), total=len(elements), disable=disable_tqdm))
    finally:
        p.close()

    return res


def all_combination_dict(suitable_var_to_pred, level):
    l = []
    for k, vals in suitable_var_to_pred.items():
        for v in vals:
            l.append(f'{k}____{v}')

    res = []
    for subset in itertools.combinations(l, level):
        d_t = {}
        for s in subset:
            key, val = s.split('____')
            if key in d_t:
                d_t[key].append(val)
            else:
                d_t[key] = [val]

        res.append(d_t)
    return res


def create_tmp_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
