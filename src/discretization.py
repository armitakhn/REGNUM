import copy
from custom_exception import NodeRulesNotWellConstructed
from rule import Rule
from dataclasses import dataclass, field
import pandas as pd
from sklearn import tree


@dataclass
class Tree:
    df: pd.DataFrame
    parent_rule: Rule
    min_event: float
    min_not_event: float
    min_pca_conf: float
    clf: tree
    feature_names: list
    id_to_node: dict = field(default_factory=dict)
    useful_nodes: list = field(default_factory=list)

    def __post_init__(self):
        self.build_tree()

    def build_tree(self):
        tree = self.clf.tree_
        # Build tree nodes
        tree_nodes = []
        for i in range(tree.node_count):
            tree_nodes.append({i: Node(tree, self.feature_names, i)})

        hope_in_nodes = [0]
        for i, n in enumerate(tree_nodes):
            if i not in hope_in_nodes:
                continue

            if not n[i].is_leaf:
                possible_left_id = n[i].left_child_id
                possible_right_id = n[i].right_child_id
                possible_left = tree_nodes[possible_left_id]
                possible_right = tree_nodes[possible_right_id]

                if possible_left[possible_left_id].n_event > self.min_event or possible_left[
                    possible_left_id].n_nonevent > self.min_not_event:  # relaxed condition
                    n['left'] = possible_left
                    n['left'][possible_left_id].update_rule(n[i].threshold_left, n[i].rule_node)
                    n['left'][possible_left_id].parent_i = n[i].parent_i.copy()
                    n['left'][possible_left_id].parent_i.append(n[i].i)
                    n['left'][possible_left_id].update_measures(self.df, self.parent_rule, self.min_pca_conf,
                                                                self.min_event)
                    if n['left'][possible_left_id].event_not_event_corrected[1] > self.min_event or \
                            n['left'][possible_left_id].event_not_event_corrected[0] > self.min_not_event:

                        n['left'][possible_left_id].set_is_useful(True)
                        hope_in_nodes.append(possible_left_id)
                    else:
                        del n["left"]

                if possible_right[possible_right_id].n_event > self.min_event or possible_right[
                    possible_right_id].n_nonevent > self.min_not_event:
                    n['right'] = possible_right
                    n['right'][possible_right_id].update_rule(n[i].threshold_right, n[i].rule_node)
                    n['right'][possible_right_id].parent_i = n[i].parent_i.copy()  # .append(n["node"].i)
                    n['right'][possible_right_id].parent_i.append(n[i].i)
                    n['right'][possible_right_id].update_measures(self.df, self.parent_rule, self.min_pca_conf,
                                                                  self.min_event)

                    if n['right'][possible_right_id].event_not_event_corrected[1] > self.min_event or \
                            n['right'][possible_right_id].event_not_event_corrected[0] > self.min_not_event:
                        n['right'][possible_right_id].set_is_useful(True)
                        hope_in_nodes.append(possible_right_id)

                    else:
                        del n['right']

                if 'left' not in n and 'right' not in n:
                    n[i].set_is_leaf(True)

            if n[i].is_useful and n[i].is_leaf:
                self.useful_nodes.append(n[i])

        for i in range(len(tree_nodes)):
            self.id_to_node[i] = tree_nodes[i][i]

    def find_nodes_rules(self, criteria="pcaBodySize", resolve_redundancy=True):
        include_useful = self._find_nodes_rules(criteria, 'include', resolve_redundancy)
        exclude_useful = self._find_nodes_rules(criteria, 'exclude', resolve_redundancy)
        res_final_nodes = {'include': [self.id_to_node[n_id] for n_id in include_useful],
                           'exclude': [self.id_to_node[n_id] for n_id in exclude_useful]}

        return res_final_nodes

    def _find_nodes_rules(self, criteria, include_exclude, resolve_redundancy):
        nodes_ok = set()
        resolve_dict = {}
        for n in self.useful_nodes:
            res = set()
            path = n.parent_i.copy()
            path.append(n.i)
            highest_score = 0
            best_id = None
            for j in path:
                n = self.id_to_node[j]
                rule_m = n.rule_measures[include_exclude]
                if rule_m['Sat_minhc'] and rule_m['Sat_pcaconf']:
                    if rule_m[criteria] > highest_score:
                        highest_score = rule_m[criteria]
                        best_id = j

                    res.add(j)

            nodes_ok.add(best_id)
            res.discard(best_id)
            resolve_dict[best_id] = res

        nodes_ok.discard(None)
        if resolve_redundancy:
            for node_ok, node_parent in resolve_dict.items():
                for k in node_parent:
                    nodes_ok.discard(k)

        return nodes_ok


class Node:

    def __init__(self, tree, feature_names, i):
        self.tree = tree
        self.i = i
        self.parent_i = []
        self.feature_name = "leaf" if self.tree.feature[self.i] == -2 else feature_names[self.tree.feature[self.i]]
        self.n_event = int(self.tree.value[self.i][0][1])
        self.n_nonevent = int(self.tree.value[self.i][0][0])
        self.n_records = self.tree.n_node_samples[self.i]
        self.event_rate = self.n_event / self.n_records
        self.left_child_id = self.tree.children_left[self.i]
        self.right_child_id = self.tree.children_right[self.i]
        self.threshold = self.tree.threshold[i]
        self.threshold_right = {self.feature_name: {'>': self.threshold}}
        self.threshold_left = {self.feature_name: {'<=': self.threshold}}
        self.is_useful = False
        self.is_leaf = self.compute_is_leaf()
        self.rule_node = {}
        self.rule_measures = {"include":
                                  {"support": None, "pcaBodySize": None,
                                   'headCoverage': None, 'pcaConfidence': None, 'f_score': None,
                                   'Sat_minhc': False, 'Sat_pcaconf': False},
                              "exclude":
                                  {"support": None, "pcaBodySize": None,
                                   'headCoverage': None, 'pcaConfidence': None, 'f_score': None,
                                   'Sat_minhc': False, 'Sat_pcaconf': False}}

        self.event_not_event_corrected = {0: 0, 1: 0}
        self.n_supp = None
        self.n_bdysize = None

    def __repr__(self):
        return f"id:{self.i}, n_event: {self.n_event}" \
               f", n_nonevent: {self.n_nonevent}, n_event_corrected: {self.event_not_event_corrected[1]}" \
               f", n_nonevent_corrected: {self.event_not_event_corrected[0]}" \
               f" n_records: {self.n_records}, rule: {self.rule_node}, rule measures: {self.rule_measures}"

    def compute_is_leaf(self):
        return self.left_child_id == self.right_child_id

    def set_is_leaf(self, boo):
        self.is_leaf = boo

    def set_is_useful(self, boo):
        self.is_useful = boo

    def set_supp_pca_bdysize(self, dict_val):
        self.n_supp = dict_val[1]
        self.n_bdysize = dict_val[0]

    def update_rule(self, d, d_parent):
        self.rule_node = copy.deepcopy(d_parent)
        if not self.rule_node:
            self.rule_node = d.copy()

        else:
            for k, sing_v in d.items():
                if k not in self.rule_node:
                    self.rule_node[k] = sing_v
                else:
                    for s, v in sing_v.items():
                        if s in self.rule_node[k]:
                            self.rule_node[k][s] = v
                        else:
                            self.rule_node[k][s] = v

    def update_measures(self, df, parent_rule: Rule, min_pca_conf, min_event):

        dict_include, dict_exclude = self._compute_measures_based_on_rules(df,
                                                                           parent_rule.conclusion.atom_raw_variables)

        self.event_not_event_corrected.update(dict_include)

        self.rule_measures['include']["support"] = dict_include[1] if 1 in dict_include else 0
        self.rule_measures['exclude']["support"] = dict_exclude[1] if 1 in dict_exclude else 0

        self.rule_measures['include']["pcaBodySize"] = self.rule_measures['include']["support"] + dict_include[
            0] if 0 in dict_include else 0
        self.rule_measures['exclude']["pcaBodySize"] = self.rule_measures['exclude']["support"] + dict_exclude[
            0] if 0 in dict_exclude else 0

        for incl_excl, dict_measures in self.rule_measures.items():
            self.rule_measures[incl_excl]["pcaConfidence"] = 0 if self.rule_measures[incl_excl]["pcaBodySize"] == 0 else \
                self.rule_measures[incl_excl]["support"] / self.rule_measures[incl_excl]["pcaBodySize"]

            self.rule_measures[incl_excl]["headCoverage"] = self.rule_measures[incl_excl][
                                                                "support"] / parent_rule.size_head_r

            try:
                f_score = 2 * (
                        self.rule_measures[incl_excl]["pcaConfidence"] * self.rule_measures[incl_excl][
                    "headCoverage"]) / (self.rule_measures[incl_excl]["pcaConfidence"] + self.rule_measures[incl_excl][
                    "headCoverage"])
            except ZeroDivisionError:
                f_score = 0

            self.rule_measures[incl_excl]["f_score"] = f_score

            self.rule_measures[incl_excl]["Sat_minhc"] = True if self.rule_measures[incl_excl][
                                                                     "support"] > min_event else False
            self.rule_measures[incl_excl]["Sat_pcaconf"] = True if self.rule_measures[incl_excl][
                                                                       "pcaConfidence"] > min_pca_conf else False

    def _compute_measures_based_on_rules(self, df, parent_rule_conclusion_vars):
        tmp_df = df.copy()
        for var_pred_feature, dict_rel_thr in self.rule_node.items():
            for rel, thr in dict_rel_thr.items():
                if rel == '>':
                    tmp_df = tmp_df[tmp_df[var_pred_feature] > thr]
                elif rel == "<=":
                    tmp_df = tmp_df[tmp_df[var_pred_feature] <= thr]
                else:
                    raise NodeRulesNotWellConstructed("Node rules not correctly constructed...")

        dict_include = tmp_df.groupby('label').apply(
            lambda dftmp: dftmp.drop_duplicates(subset=parent_rule_conclusion_vars).shape[0]).to_dict()

        tmp_df_exclude = df[~df.index.isin(tmp_df.index)]
        dict_exclude = tmp_df_exclude.groupby('label').apply(
            lambda dftmp_ex: dftmp_ex.drop_duplicates(subset=parent_rule_conclusion_vars).shape[0]).to_dict()

        return dict_include, dict_exclude
