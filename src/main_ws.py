from read_args import read_args
from data_loader import GeneralDataLoader
from parent_ruleminer import RunParseAMIE
from graph_data import StarDogGraph
from rule_writer import WriteNewRules
from rule_enricher_nums import RuleEnricherNumerical
import os
import conf


def main_enricher_ws(path_t, path_numerical_preds, num_atoms, min_conf, min_hc):
    args = read_args()

    dl = GeneralDataLoader(path_t=path_t, path_numerical_preds=path_numerical_preds)

    amie = RunParseAMIE(data=dl.df, path_amie3=args.path_amie,
                        path_save_rules=args.path_save_rules,
                        num_atoms=num_atoms,
                        min_conf=min_conf,
                        min_hc=min_hc)

    rules = amie.parse_amie()
    graph = StarDogGraph(dl)
    rule_loader = WriteNewRules(saving_path=args.path_save_enriched_rules)

    preds = list(dl.numerical_preds)
    dict_all_new_rules = {}
    for idx, rule in enumerate(rules[0:10]):
        ren = RuleEnricherNumerical(rule, graph, preds, merge_intervals=True, debug_mode=False)
        dict_all_new_rules[idx] = ren.list_new_rules

    return rule_loader.dump_json(dict_all_new_rules)


def prepare_path_env():
    if not os.path.exists(conf.PATH_SAVE_AMIE_RULE):
        print(conf.PATH_SAVE_AMIE_RULE)
        os.makedirs(conf.PATH_SAVE_AMIE_RULE)

