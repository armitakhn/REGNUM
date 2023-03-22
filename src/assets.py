from itertools import groupby, chain
from collections import Counter


def get_sequences(intervals):
    return [
        [x for _, x in g]
        for k, g in groupby(enumerate(intervals), lambda i_x: i_x[0] - i_x[1])
    ]


def _get_sequences(intervals):
    return [list(g) for _, g in groupby(intervals, lambda x: x[0] - x[1])]


def dict_df_two_cols(df, key1, key2):
    k1_k2_df_dict = {}
    for keys, value in df.groupby([key1, key2]):
        k1, k2 = keys
        if k1 not in k1_k2_df_dict:
            k1_k2_df_dict[k1] = {}
        k1_k2_df_dict[k1][k2] = value
    return k1_k2_df_dict


def _dict_df_two_cols(df, key1, key2):
    k1_k2_df_dict = {}
    for k1, k2, value in df.groupby([key1, key2]):
        k1_k2_df_dict.setdefault(k1, {})[k2] = value
    return k1_k2_df_dict


def dict_pred_to_df(df_train, df_test):
    predicates_train = df_train.predicate.unique()
    pred_to_df_test = {}
    for pred in predicates_train:
        pred_to_df_test[pred] = df_test[df_test.predicate == pred]
    return pred_to_df_test


def expert_conf(dict_test_rules_cands, rules):
    dict_rule_conf = {rule_num: rules[rule_num].pcaConfidence for rule_num in dict_test_rules_cands.keys()}
    highest_conf_rule = max(dict_rule_conf, key=dict_rule_conf.get)
    # print(dict_test_rules_cands[highest_conf_rule])
    random_pick = list(dict_test_rules_cands[highest_conf_rule])[0]
    return random_pick


def democracy(dict_test_rules_cands, rules=None):
    flattened_candidates = list(chain(*dict_test_rules_cands.values()))
    value, count = Counter(flattened_candidates).most_common(1)[0]
    return value


"""
def democracy(dict_test_rules_cands, rules=None):
    flattened_candidates = list(chain(*dict_test_rules_cands.values()))
    if not flattened_candidates:
        return None, 0
    counter = Counter(flattened_candidates)
    most_common = counter.most_common(1)
    if len(most_common) == 1:
        return most_common[0]
    else:
        return None, 0

"""
