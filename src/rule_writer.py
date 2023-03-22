from enum import Enum
from conf_amie import MIN_HC, MIN_CONF
import json
import numpy as np
from rule import Rule, Atom


class SavingModeRules(Enum):
    JSON = "json"
    AMIE_LIKE = "amie-like"
    JSON_BENCHMARK = "json-bnc"


def dump_json(dict_all_new_rules, f):
    return json.dumps(dict_all_new_rules, f, indent=4, cls=CustomJsonEncoder)


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Atom):
            return str(obj)
        if isinstance(obj, Rule):
            return obj.rule
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(CustomJsonEncoder, self).default(obj)


class WriteNewRules:
    def __init__(self, saving_path, saving_mode=SavingModeRules.JSON, binning="mdlp"):
        self.saving_path: str = saving_path
        self.saving_mode: SavingModeRules = saving_mode
        self.binning_technique = binning
        # self._meta_data_writer()

    def _meta_data_writer(self):
        if self.saving_mode.value == "amie-like":
            self._meta_data_rules()

    def write_numerical_pred_rules(self, dict_all_new_rules):
        if self.saving_mode.value == "amie-like":
            self._write_numerical_pred_rules_amie_like(dict_all_new_rules)
        if self.saving_mode.value == "json":
            # not useful anymore
            self._write_numerical_pred_rules_json(dict_all_new_rules)
        if self.saving_mode.value == "json-bnc":
            dict_kgc = self._bnc_rule_creation(dict_all_new_rules)
            self._write_numerical_pred_rules_json(dict_kgc)

    def _bnc_rule_creation(self, _dict_all_new_rules):  # dict_all_new_rules
        binnings_str = list(_dict_all_new_rules[0].keys())
        print(binnings_str)
        res_ = {}
        for bin_ in binnings_str:
            dict_kgc = {}
            for rule_idx, rule_list in _dict_all_new_rules.items():
                try:
                    dict_kgc[rule_idx] = {'parent_rule': rule_list[bin_][0]['parent_rule'], 'var_pred': {}}
                except:
                    print(rule_idx)
                    print(rule_list)
                    raise Exception()
                for rl in rule_list[bin_]:
                    if 'var_num' in rl:
                        if rl['var_num'] not in dict_kgc[rule_idx]['var_pred']:
                            dict_kgc[rule_idx]['var_pred'][rl['var_num']] = {}
                            dict_kgc[rule_idx]['var_pred'][rl['var_num']][rl['pred']] = [rl]

                        else:
                            if rl['pred'] not in dict_kgc[rule_idx]['var_pred'][rl['var_num']]:
                                dict_kgc[rule_idx]['var_pred'][rl['var_num']][rl['pred']] = [rl]

                            else:
                                dict_kgc[rule_idx]['var_pred'][rl['var_num']][rl['pred']].append(rl)
            res_[bin_] = dict_kgc
        return res_

    def _meta_data_rules(self):
        with open(self.saving_path, "w") as f:
            f.write(f"Adding numerical predicates to rules mined by AMIE \n")
            f.write(f"Using HeadCoverage as pruning metric with minimum threshold {MIN_HC} \n")
            f.write(f"Filtering on PCA confidence with minimum threshold {MIN_CONF} \n")
            cols = "Rule HeadCoverage	StdConfidence	PCAConfidence  Support	BodySize	PCABodysize	" \
                   "FunctionalVariable  PredicateNumerical    NumericalVariable BeginInterval   EndInterval "
            f.write(cols)
            f.write('\n')
            f.close()

    @staticmethod
    def create_str_new_rule(parent_rule, dict_new_rule):
        # TODO: later put the preds, var, intervals in a list and loop over them for creating the string of the rule
        new_rule = f'{dict_new_rule["var_num"]} {dict_new_rule["pred"]} ?num  ?num  not-between  [{dict_new_rule["beginInterval"]}, {dict_new_rule["endInterval"]}] '
        return new_rule + parent_rule.rule

    @staticmethod
    def _write_original_rule(parent_rule, f):
        f.write(parent_rule.line + '\n')

    def _write_numerical_pred_rules_amie_like(self, dict_all_new_rules):
        with open(self.saving_path, "a") as f:
            for idx, dict_bin_rules in dict_all_new_rules.items():
                for binning, lis_parent_rule in dict_bin_rules.items():
                    if binning != self.binning_technique:
                        continue

                    _tmp_l = lis_parent_rule[0]
                    self._write_original_rule(_tmp_l['parent_rule'], f)
                    # print(lis_parent_rule)
                    if len(lis_parent_rule) == 1:
                        for _ in range(5):
                            f.write('\n')
                        continue
                    for dict_new_rule in lis_parent_rule[1:]:
                        # rule = self.create_str_new_rule(dict_new_rule['parent_rule'], dict_new_rule)
                        enriched_rule = dict_new_rule['enriched_rule']
                        headCoverage = dict_new_rule['headCoverage']
                        stdConfidence = dict_new_rule['stdConfidence']
                        pcaConfidence = dict_new_rule['pcaConfidence']
                        # f_score = dict_new_rule['f_score']
                        support = dict_new_rule['support']
                        bodySize = dict_new_rule['bodySize']
                        pcaBodysize = dict_new_rule['pcaBodySize']
                        functionalVariable = dict_new_rule['functionalVariable']
                        predicateNumerical = dict_new_rule['pred']
                        numericalVariable = dict_new_rule['var_num']
                        beginInterval = dict_new_rule['beginInterval']
                        endInterval = dict_new_rule['endInterval']
                        # include_exclude = dict_new_rule['include_exclude']

                        f.write(
                            f'{enriched_rule}\t{headCoverage}\t{stdConfidence}\t{pcaConfidence}\t{support}\t{bodySize}\t{pcaBodysize}\t{functionalVariable}\t{predicateNumerical}\t{numericalVariable}\t{beginInterval}\t{endInterval}\n')
                    for _ in range(5):
                        f.write('\n')

    def _write_numerical_pred_rules_json(self, dict_all_new_rules):
        with open(self.saving_path, "w") as file:
            json.dump(dict_all_new_rules, file, indent=4, cls=CustomJsonEncoder)

    @staticmethod
    def dump_json(dict_all_new_rules):
        return json.dumps(dict_all_new_rules, indent=4, cls=CustomJsonEncoder)
