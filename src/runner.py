from rule_enricher_nums import RuleEnricherNumerical as ren
from time import time
from tqdm import tqdm


def run(rules, gr, preds):
    dict_all_new_rules_f_score = {}

    for idx, rule in enumerate(tqdm(rules)):
        try:
            rren = ren(rule=rule, RDFGraph=gr, numerical_preds=preds, max_atom_num=4, debug_mode=False,
                       add_num_pred_grounded_not_functional=False, only_margin_rules=True, exact_cut=True)
            dict_all_new_rules_f_score[idx] = {'parent_rule': rule}

            dict_all_new_rules_f_score[idx]['numerical_rules'] = [n.build_dict() for n in
                                                                  rren.enriched_rules['f_score']]

        except Exception as e:
            print(idx, e)

    return dict_all_new_rules_f_score
