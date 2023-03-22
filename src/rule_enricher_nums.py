from rule import Rule
from conf_amie import MIN_HC
from graph_query import query_rule_num_pred, query_rule_add_base, query_select_useful_preds, query_main_df
import numpy as np
from utils import all_combination_dict
import random
from sklearn import tree
from rule_enriched import ConstructRules
from discretization import Tree
from time import time


class BaseRuleEnricherNumerical:
    def __init__(self, rule, RDFGraph, numerical_preds, minhc=MIN_HC,
                 add_num_pred_grounded_not_functional=False, only_margin_rules=False, debug_mode=False):
        self.rule: Rule = rule
        self.graph = RDFGraph
        self.numerical_preds: list = numerical_preds
        self.only_margin_rules = only_margin_rules
        self.margin_conf = 0.2  # at least by % better than parent conf
        self.margin_hc = 0.2  # at most % worse than hc
        self.min_conf_pr = self.compute_min_conf()
        self.even_not_func_enrich_g: bool = add_num_pred_grounded_not_functional
        self.debug_mode: bool = debug_mode
        self.dict_pred_var_df_debug = {}  # only for debug
        self._base_query_df, self._rep_var = query_rule_add_base(self.rule, self.even_not_func_enrich_g)
        self.min_pos_example_satisfy_minhc = self._compute_min_pos_example_satisfy_minhc()
        self.suitable_var_to_pred, self.var_pred_to_relaxed_supp = self._compute_var_to_pred_relaxed_supp()
        self.max_per_level = 300

    def _compute_min_pos_example_satisfy_minhc(self):
        hard_pos_ex = (1 - self.margin_hc) * self.rule.headCoverage * self.rule.size_head_r
        return hard_pos_ex

    def compute_min_conf(self):
        if not self.only_margin_rules:

            self.min_conf_pr = self.rule.pcaConfidence
        else:
            self.min_conf_pr = (1 + self.margin_conf) * self.rule.pcaConfidence
        return self.min_conf_pr

    def _find_useful_pred_per_var(self, var):
        qq = query_select_useful_preds(self.rule, var)
        useful_preds = self.graph.query_dataframe(qq)

        return set(useful_preds.useful_preds).intersection(set(self.numerical_preds))

    def _compute_var_to_pred_relaxed_supp(self):
        var_to_pred_list = {}
        var_pred_to_relaxed_supp = {}

        for var in self.rule.rule_variables:
            possible_num_preds = self._find_useful_pred_per_var(var)
            for pred in possible_num_preds:  # self.numerical_preds:
                plaus, relaxed_supp = self._checker_relaxed_supp({var: [pred]})
                if plaus:  # relaxed head coverage condition not satisfied...
                    if var in var_to_pred_list:
                        var_to_pred_list[var].append(pred)
                        var_pred_to_relaxed_supp[var][pred] = relaxed_supp
                    else:
                        var_to_pred_list[var] = [pred]
                        var_pred_to_relaxed_supp[var] = {pred: relaxed_supp}

        return var_to_pred_list, var_pred_to_relaxed_supp

    def _checker_relaxed_supp(self, var_preds_dict):
        # query - count --> relaxed_supp
        qs = query_rule_num_pred(self.rule, var_preds_dict)
        #######
        # it's ok
        relaxed_supp = self.graph.query_count(qs)

        plaus = self._checker_satisfy_minhc(relaxed_supp)
        if self.debug_mode and plaus:
            print(f"the query to compute the relaxed supp: \n {qs}")
            # print(f"relaxed supp: \n {relaxed_supp}")
            # print("ok..\n")
            # print(qs)

        return plaus, relaxed_supp

    def _checker_satisfy_minhc(self, new_supp):
        return True if new_supp > self.min_pos_example_satisfy_minhc else False

    def _checker_satisfy_minconf(self, new_pcaconf):
        return True if new_pcaconf > self.min_conf_pr else False

    def relaxed_supp_level(self, level):
        comb_num = 0
        relaxed_supp_level_l = []
        for r in all_combination_dict(self.suitable_var_to_pred, level):
            plaus, relaxed_supp = self._checker_relaxed_supp(r)  # is relaxed_supp same as existential support? yes!
            if plaus:
                comb_num += 1
                relaxed_supp_level_l.append((r, relaxed_supp))
            if comb_num > self.max_per_level:
                return relaxed_supp_level_l

        return relaxed_supp_level_l

    def query_get_df(self, var_preds_dict):
        q_out_str, dict_feature_var_pred = query_main_df(self._base_query_df, var_preds_dict)
        df = self.graph.query_dataframe(q_out_str)
        df['label'] = np.where((df['anything'] == df[self._rep_var[1:]]), 1, 0)
        df = df.drop(['anything'], axis=1)
        df = df.drop_duplicates()
        cols = list(df.columns)
        cols.remove('label')
        idxs = []
        for uu, df__ in df.groupby(cols):
            if df__.shape[0] > 1:
                idxs.extend(list(df__[df__['label'] == 0].index))

        df.loc[idxs, 'label'] = 1
        df = df.drop_duplicates()
        # remove_cols = set(cols) - set(dict_feature_var_pred.keys())
        # df = df.drop(list(remove_cols), axis=1)

        df = df.rename(columns=dict_feature_var_pred)
        return df, dict_feature_var_pred

    def _delete_preds_diversity(self, d):
        for var, preds in d.items():
            for pred in preds:
                if pred in self.suitable_var_to_pred[var]:
                    self.suitable_var_to_pred[var].remove(pred)


class RuleEnricherNumerical(BaseRuleEnricherNumerical):

    def __init__(self, rule, RDFGraph, numerical_preds, max_atom_num, debug_mode, add_num_pred_grounded_not_functional,
                 only_margin_rules, exact_cut):
        super().__init__(rule=rule, RDFGraph=RDFGraph, numerical_preds=numerical_preds,
                         add_num_pred_grounded_not_functional=add_num_pred_grounded_not_functional,
                         only_margin_rules=only_margin_rules, debug_mode=debug_mode)
        self.max_atom_num = max_atom_num
        self.exact_cut = exact_cut
        self._dict_df_pred_var = {}
        self.enriched_rules = {'f_score': [], 'pcaBodySize': []}
        self.run()

    def compute_min_sample_leaf(self, rs_supp, rs_pca_bdysize):
        min_event = int(self.min_pos_example_satisfy_minhc)
        min_not_event = int(rs_pca_bdysize - (rs_supp / self.min_conf_pr))
        return min_event, min_not_event

    def fit_model(self, x, y, min_sample_l):
        clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=min_sample_l + 1)
        clf.fit(x, y)
        return clf


    def _find_existential_num_(self, rs_pca_bdysize, rs_supp):
        pca_body_size_existential = rs_pca_bdysize
        support_existential = rs_supp
        conf_existential = support_existential / pca_body_size_existential
        has_existential = conf_existential > self.min_conf_pr
        pca_conf_existential = support_existential / pca_body_size_existential
        return has_existential, pca_body_size_existential, support_existential, pca_conf_existential

    def run(self):
        level = 1
        while sum((len(d) for d in self.suitable_var_to_pred.values())) > level and level < 4:
            self.run_l(level)
            level += 1

    def run_l(self, level):
        if self.min_pos_example_satisfy_minhc == 1:
            return None
        t1 = time()

        relaxed_supp_level_ = self.relaxed_supp_level(level)
        t2 = time()

        random.shuffle(relaxed_supp_level_)

        for d, rs_supp in relaxed_supp_level_:
            # if d !={'?b': ['http://patronage_1'], '?a': ['http://patronage_1']}:
            #    continue
            # print(level, rs_supp, self.min_pos_example_satisfy_minhc, d)
            t1 = time()

            create_rule_combi = False

            df, self.dict_feature_var_pred = self.query_get_df(d)
            t2 = time()
            # print(f't2-t1 {t2-t1}')

            rs_pca_bdysize = df.drop_duplicates(self.rule.conclusion.atom_raw_variables).shape[0]
            t3 = time()
            # print(f't3-t2 {t3-t2}')

            min_event, min_not_event = self.compute_min_sample_leaf(rs_supp, rs_pca_bdysize)
            if min_not_event < 2:
                continue
            t4 = time()
            # print(f't4-t3 {t4-t3}')

            # Existential with the numerical predciates
            has_existential, pca_body_size_existential, support_existential, pca_conf_existential = self._find_existential_num_(
                rs_pca_bdysize, rs_supp)
            t5 = time()
            # print(f't5-t4 {t5-t4}')

            if has_existential:
                include_exclude = 'existential'
                rc = ConstructRules(self.rule, d, support_existential, pca_body_size_existential,
                                    pca_conf_existential, include_exclude)
                self.enriched_rules['pcaBodySize'].append(rc)
                self.enriched_rules['f_score'].append(rc)

                # print(rc.build_dict())

                create_rule_combi = True

            t6 = time()
            # print(f't6-t5 {t6-t5}')

            X, y = df[self.dict_feature_var_pred.values()], df['label']
            self.X = X
            self.y = y

            clf = self.fit_model(X, y, min(min_event, min_not_event))
            if clf.tree_.node_count == 1:
                continue
            t7 = time()
            # print(f't7-t6 {t7-t6}')

            tree = Tree(df=df, parent_rule=self.rule, min_event=min_event, min_not_event=min_not_event,
                        min_pca_conf=self.min_conf_pr, clf=clf, feature_names=X.columns)

            useful_dict_pcaBodySize = tree.find_nodes_rules(criteria='pcaBodySize', resolve_redundancy=True)
            useful_dict_f_score = tree.find_nodes_rules(criteria='f_score', resolve_redundancy=True)

            t8 = time()
            # print(f't8-t7 {t8-t7}')

            if len(useful_dict_f_score["include"]) > 0 or len(useful_dict_f_score["exclude"]) > 0:
                create_rule_combi = True

            self._add_rule(level, useful_dict_pcaBodySize, 'pcaBodySize')
            self._add_rule(level, useful_dict_f_score, 'f_score')

            t9 = time()

            if create_rule_combi:
                self._delete_preds_diversity(d)

            # print(f't9-t8 {t9-t8}')

            # print(f'total {t9-t1}')
    def _add_rule(self, level, useful_dict, name):
        for include_exclude, nodes in useful_dict.items():
            for node in nodes:
                head_coverage = node.rule_measures[include_exclude]['headCoverage']
                support = node.rule_measures[include_exclude]['support']
                pca_body_size = node.rule_measures[include_exclude]['pcaBodySize']
                pca_conf = node.rule_measures[include_exclude]['pcaConfidence']
                f_score = node.rule_measures[include_exclude]['f_score']

                rc = ConstructRules(rule=self.rule, numerical_part=node.rule_node, support=support,
                                    pca_body_size=pca_body_size, pca_confidence=pca_conf,
                                    include_exclude=include_exclude,
                                    head_coverage=head_coverage, f_score=f_score, level=level)

                self.enriched_rules[name].append(rc)
