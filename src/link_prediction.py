from dataclasses import dataclass, field
from abc import abstractmethod
from graph_query import create_query_kg_completion, \
    create_query_kg_completion_numerical
from enum import Enum
import pandas as pd
import tqdm
from graph_data import RDFGraph
from assets import democracy, expert_conf, dict_df_two_cols, dict_pred_to_df


class Wrapper:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class AggregatingStrategies(Enum):
    DEMOCRACY = Wrapper(democracy)
    EXPERT = Wrapper(expert_conf)
    CONF_WEIGHT = "confidence_weight"  # TODO: also use the aggreagate from the ESWC22 paper- NN


@dataclass
class BasebenchmarkKGC:
    graph: RDFGraph
    df_test: pd.DataFrame
    df_train: pd.DataFrame  # df.dl from dataloader
    rules: field(default_factory=list) = None
    kg_completion_task: str = "head"  # if head --? (?, p, o) if tail : (s, p ,?)
    rules_contain_conts: bool = False

    def __post_init__(self):
        self.set_entity_based_on_kg_task()
        self.test_to_responses_dict = {key: {} for key in list(range(self.df_test.shape[0]))}
        self.dict_df_pred_obj = dict_df_two_cols(self.df_train, "predicate", self.entity_known)
        self.pred_to_df_test = dict_pred_to_df(self.df_train, self.df_test)
        self.aggregate_dict_to_func = {"democ": self.popul_democ, "max": self.popul_max,
                                       "weightedfscore": self.popul_ws, "noisy-or": self.popul_noisy_or}

    def set_entity_based_on_kg_task(self):
        if self.kg_completion_task == "head":
            self.entity_known = "object"
            self.entity_uk = "subject"
            self.head_missing = True
        elif self.kg_completion_task == "tail":
            self.entity_known = "subject"
            self.entity_uk = "object"
            self.head_missing = False

    def hits_at_k(self, mode="democ", k=10):
        # print(f"computing hits@{k}...")
        correct_pred = 0
        test_exist = 0
        for test_idx, dict_test_rules_cands in self.test_to_responses_dict.items():
            # print(test_idx)
            # print(dict_test_rules_cands)
            if not dict_test_rules_cands:
                l_res = []
            else:
                l_res = self._create_dict_cand_to_score(mode, dict_test_rules_cands, k)

            if self.df_test[self.entity_uk].iloc[test_idx] in l_res:
                correct_pred += 1

            if len(l_res) > 0:
                test_exist += 1

        return (correct_pred / self.df_test.shape[0]) * 100, (correct_pred / test_exist) * 100

    def _create_dict_cand_to_score(self, mode, dict_test_rule_cands, k=10):
        if mode == "weightedfscore":
            self.dict_weighted_fscore = {rule_id: 1 / len(cands) * self.map_idx_rule[rule_id]['f_score'] for
                                         rule_id, cands in dict_test_rule_cands.items()}
        dict_cand_to_score = {}
        for rule_id, cands in dict_test_rule_cands.items():
            for cand in cands:
                ##print(dict_cand_to_score)

                dict_cand_to_score = self.aggregate_dict_to_func[mode](dict_cand_to_score, cand, rule_id)
        # print("dict_cand_to_score", dict_cand_to_score)
        if mode == "noisy-or":
            dict_cand_to_score = {cand: 1 - score for cand, score in dict_cand_to_score.items()}

        dict_cand_to_score = dict(sorted(dict_cand_to_score.items(), key=lambda item: item[1], reverse=True))
        # print("final", dict_cand_to_score)
        return list(dict_cand_to_score.keys())[:k]

    def popul_democ(self, dict_cand_to_score, cand, rule_id=None):
        if cand not in dict_cand_to_score:
            dict_cand_to_score[cand] = 1
        else:
            dict_cand_to_score[cand] += 1
        return dict_cand_to_score

    def popul_max(self, dict_cand_to_score, cand, rule_id):
        if cand not in dict_cand_to_score:
            dict_cand_to_score[cand] = self.map_idx_rule[rule_id]['pca_confidence']
        else:
            # print(self.map_idx_rule[rule_id]['pcaConfidence'])
            # print(dict_cand_to_score[cand])
            dict_cand_to_score[cand] = max(self.map_idx_rule[rule_id]['pca_confidence'], dict_cand_to_score[cand])
        return dict_cand_to_score

    def popul_ws(self, dict_cand_to_score, cand, rule_id):
        if cand not in dict_cand_to_score:
            dict_cand_to_score[cand] = self.dict_weighted_fscore[rule_id]
        else:
            dict_cand_to_score[cand] += self.dict_weighted_fscore[rule_id]  ##can be changed to average
        return dict_cand_to_score

    def popul_noisy_or(self, dict_cand_to_score, cand, rule_id):
        if cand not in dict_cand_to_score:
            dict_cand_to_score[cand] = 1 - self.map_idx_rule[rule_id]['pca_confidence']
        else:
            dict_cand_to_score[cand] *= 1 - self.map_idx_rule[rule_id]['pca_confidence']
        return dict_cand_to_score

    @abstractmethod
    def link_pred_test_to_dict(self):
        pass

    def _check_con_instantiated_corr(self, rule, inst_possible):

        if self.entity_known == "object":
            if rule.conclusion.objectD not in rule.rule_variables:
                if rule.conclusion.objectD[1:-1] != inst_possible:
                    return True
        elif self.entity_known == "subject":
            if rule.conclusion.subject not in rule.rule_variables:
                if rule.conclusion.subject[1:-1] != inst_possible:  # TODO use subjectraw
                    return True
        return False


@dataclass
class BenchmarkKGCAMIE(BasebenchmarkKGC):
    useful_rules: list = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.map_idx_rule = {i: self.rules[i].toDict() for i in range(len(self.rules))}
        self.link_pred_test_to_dict()

    def link_pred_test_to_dict(self):
        # idx_rules = []
        # for idx, rule in enumerate(tqdm.tqdm(self.rules)):
        #    idx_rules.append((idx, rule))

        # mult_proc_apply(partial(self._link_pred_query_populate), idx_rules)
        for idx, rule in enumerate(tqdm.tqdm(self.rules)):
            if '!=' in rule.rule:
                continue
            if idx not in self.useful_rules and len(self.useful_rules) > 0:
                continue

            self._link_pred_query_populate((idx, rule))

    def _link_pred_query_populate(self, idx_rule):
        idx, rule = idx_rule
        curr_p = rule.conclusion.predicate_raw
        curr_df_test = self.pred_to_df_test[curr_p]

        if curr_df_test.shape[0] == 0:
            return
        insts_possible = curr_df_test[self.entity_known].unique()
        dict_inst_res_tmp = {}  # only for faster execution - not to have redundant queries

        query, var_name, instance = create_query_kg_completion(rule, insts_possible, head_missing=self.head_missing)
        # print(query)
        # print(instance)
        df_ = self.graph.query_dataframe(query)
        # print(df_)

        inst_to_list = df_.groupby(instance)[var_name].apply(list).to_dict()

        for inst_possible in insts_possible:
            if self.rules_contain_conts and self._check_con_instantiated_corr(rule,
                                                                              inst_possible):  # todo: check if rules do not contain constants, skip.
                dict_inst_res_tmp[inst_possible] = []
                continue
            else:
                if inst_possible in inst_to_list:
                    dict_inst_res_tmp[inst_possible] = inst_to_list[inst_possible]
                else:
                    dict_inst_res_tmp[inst_possible] = []

        for k, row in curr_df_test.iterrows():
            instance_kn = row[self.entity_known]

            if len(dict_inst_res_tmp[instance_kn]) > 0:
                ## FILTERING
                try:
                    to_filter_in_train_p_obj = set(
                        self.dict_df_pred_obj[curr_p][instance_kn][self.entity_uk].unique())
                except:
                    to_filter_in_train_p_obj = set()

                filtered_res = set(dict_inst_res_tmp[instance_kn]) - to_filter_in_train_p_obj
                if len(filtered_res) > 0:
                    self.test_to_responses_dict[k][idx] = filtered_res


@dataclass
class BenchmarckNumKGC(BasebenchmarkKGC):
    dict_all_new_rules: dict = None

    def __post_init__(self):
        super().__post_init__()
        self.map_idx_rule = {}
        self.link_pred_test_to_dict()

    def __sum_dicts(self, d1, d2):
        d1.update(d2)
        return d1

    def _sum_dicts(self, d1_amie, d_numamie, concat_all):
        comb_d = {}
        for test_id in range(len(d1_amie)):
            if len(d_numamie[test_id]) == 0 or concat_all:
                d_amie = self.__sum_dicts(d1_amie[test_id], d_numamie[test_id])
            else:
                d_amie = d_numamie[test_id]
            comb_d[test_id] = d_amie

        return comb_d

    def combine_amie_numamie(self, only_useful=False, concat_all=True):
        self.useful_rules = []
        if only_useful:
            for rule_num_r, dict_v in (self.dict_all_new_rules.items()):
                dict_var_to_rules = dict_v["numerical_rules"]
                if len(dict_var_to_rules) > 0:
                    self.useful_rules.append(int(rule_num_r))

        bkgamie = BenchmarkKGCAMIE(useful_rules=self.useful_rules, graph=self.graph, df_test=self.df_test,
                                   df_train=self.df_train, rules=self.rules,
                                   kg_completion_task=self.kg_completion_task)

        self.test_to_responses_dict = self._sum_dicts(d1_amie=bkgamie.test_to_responses_dict,
                                                      d_numamie=self.test_to_responses_dict, concat_all=concat_all)
        self.map_idx_rule = self.__sum_dicts(bkgamie.map_idx_rule, self.map_idx_rule)

    @property
    def enriched_rules(self):
        return self.useful_rules

    def link_pred_test_to_dict(self):
        """
        TODO: for loop over rule, for each rule, find all possible cands for the test data that
        shares a predicte with rule.conclusion.predicate
        keep a global dictionary
        update for each test point the possible candidates
        """
        self.useful_rules = []
        self.test_productive = set()
        for rule_idx, one_pr_enriched_dict in tqdm.tqdm(self.dict_all_new_rules.items()):

            parent_rule = one_pr_enriched_dict['parent_rule']
            one_rule_enriched_list = one_pr_enriched_dict["numerical_rules"]

            if len(one_rule_enriched_list) > 0:
                self.useful_rules.append(int(rule_idx))

            curr_p = parent_rule.conclusion.predicate_raw
            curr_df_test = self.pred_to_df_test[curr_p]
            if curr_df_test.shape[0] == 0:
                continue

            insts_possible = curr_df_test[self.entity_known].unique()
            dict_inst_res_tmp = {}

            for enum_enrich, enriched_dict in enumerate(one_rule_enriched_list):
                include_exclude = enriched_dict['include_exclude']
                numerical_part = enriched_dict['numerical_part']

                query, var_name, instance = create_query_kg_completion_numerical(parent_rule=parent_rule,
                                                                                 numerical_part_dict=numerical_part,
                                                                                 include_exclude=include_exclude,
                                                                                 instances=insts_possible,
                                                                                 head_missing=self.head_missing)


                #if len(numerical_part)==2 and include_exclude=='include':
                #    print(query)
                #    print(numerical_part)
                #    print(include_exclude)

                df_ = self.graph.query_dataframe(query)

                inst_to_list = df_.groupby(instance)[var_name].apply(list).to_dict()

                for inst_possible in insts_possible:
                    if self.rules_contain_conts and self._check_con_instantiated_corr(parent_rule,
                                                                                      inst_possible):
                        dict_inst_res_tmp[inst_possible] = []
                        continue
                    else:
                        if inst_possible in inst_to_list:
                            dict_inst_res_tmp[inst_possible] = inst_to_list[inst_possible]
                        else:
                            dict_inst_res_tmp[inst_possible] = []

                for k, row in curr_df_test.iterrows():
                    instance_kn = row[self.entity_known]

                    if len(dict_inst_res_tmp[instance_kn]) > 0:
                        ## FILTERING
                        try:
                            to_filter_in_train_p_obj = set(
                                self.dict_df_pred_obj[curr_p][instance_kn][self.entity_uk].unique())

                        except:
                            to_filter_in_train_p_obj = set()

                        filtered_res = set(dict_inst_res_tmp[instance_kn]) - to_filter_in_train_p_obj
                        if len(filtered_res) > 0:
                            self.test_to_responses_dict[k][(rule_idx, enum_enrich)] = filtered_res
                            if (rule_idx, enum_enrich) not in self.map_idx_rule:
                                self.test_productive.add(k)
                                self.map_idx_rule[(rule_idx, enum_enrich)] = enriched_dict

                        # print(self.test_to_responses_dict)