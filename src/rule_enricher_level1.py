from graph_query import query_rule_add_df
from binning import OptimalBinningFitStats, MDLPFitStats, TreeBinnig, UnsupervisedBinning, CustomMDLPFitStats
from quality_measures import MeasureEnriched, MeasureExistential
from rule_enricher_nums import BaseRuleEnricherNumerical
from assets import get_sequences
import numpy as np
from matplotlib import pyplot as plt
from time import time


class EnrichLevel1(BaseRuleEnricherNumerical):

    def __init__(self, rule, RDFGraph, numerical_preds, binningtechniques=['mdlp', 'opt'], merge_intervals=False,
                 merge_all_intervals=False, debug_mode=False, add_num_pred_grounded_not_functional=False,
                 only_margin_rules=True):

        super().__init__(rule=rule, RDFGraph=RDFGraph, numerical_preds=numerical_preds, debug_mode=debug_mode
                         , add_num_pred_grounded_not_functional=add_num_pred_grounded_not_functional,
                         only_margin_rules=only_margin_rules)

        self.binningtechniques = binningtechniques
        self.merge_bc_intervals: bool = merge_intervals
        self.merge_all_intervals: bool = merge_all_intervals
        self.dict_bin_new_rules = {bin_teq: [{"parent_rule": self.rule}] for bin_teq in binningtechniques}  # result
        self.imbalanced_perc = []
        self.rule_enricher()

    def _check_not_func_grounded(self):
        if self.rule.functionalVariable == self.rule.conclusion.subject:
            if self.rule.conclusion.objectD not in self.rule.rule_variables:
                return False
        else:
            if self.rule.conclusion.subject not in self.rule.rule_variables:
                return False
        return True

    def rule_enricher(self):

        # TODO: if self.suitable_var_to_pred has all its values as an empty list, skip
        if not self.even_not_func_enrich_g:  # check for the functionality of the rule. If it's grounded, skip it
            if not self._check_not_func_grounded():

                if self.debug_mode:
                    print("functional grounded not proceeding...")
                return None

        for var, preds in self.suitable_var_to_pred.items():
            if not preds:
                continue
            self.add_numerical_pred_to_rule(var, preds)

    def _create_dict_df_pred_var(self, pred, var, feat_pd_):
        if pred not in self.dict_pred_var_df_debug:  # TODO: remove this later
            self.dict_pred_var_df_debug[pred] = {}
        self.dict_pred_var_df_debug[pred][var] = feat_pd_

    def _checker_better_that_parent_confidence(self, event_rate):
        # not_event_interval =  binning.n_nonevent[interval_inx]
        # event_interval =  binning.n_event[interval_inx]
        # not_event_interval/(not_event_interval+ event_interval)
        return True if self.rule.pcaConfidence > event_rate else False

    def _merge_biggest_consecutive_intervals_binning_dict(self, binning, df,
                                                          binning_dict):  # TODO: fix this! buggy, include_exclude
        intervals_seq = get_sequences(sorted(binning_dict.keys()))  # [[1,2,3], [8,9]]
        inc_ex_set = set()
        for intervals_ in intervals_seq:
            for interval_ in intervals_:
                inc_ex_set.add(binning_dict[interval_]['include_exclude'])
        if len(inc_ex_set) > 1:
            if self.debug_mode:
                print("include exclude mixed not possible for merging")
            return binning_dict
        include_exclude = list(inc_ex_set)[0]

        intervals_seq = [seq for seq in intervals_seq if len(seq) > 1]
        if not intervals_seq:
            if self.debug_mode:
                print("binning dict not empty but no consecutive intervals seq")
            return binning_dict
        tmp_ = binning_dict[list(binning_dict.keys())[0]]
        var, pred = tmp_["var_num"], tmp_["pred"]
        for interval_seq in intervals_seq:
            b_sort_idx_l = binning.sort_indx.tolist()
            # print("binning.bins_intervals[interval_seq[0]][0]",
            #      binning.bins_intervals[b_sort_idx_l.index(interval_seq[0])][0])
            begin, end = binning.bins_intervals[b_sort_idx_l.index(interval_seq[0])][0], \
                         binning.bins_intervals[b_sort_idx_l.index(interval_seq[-1])][1]

            ## recalcualte event rate
            _, _, event_rate_int = binning.recompute_stats_new_interval(interval_seq[0],
                                                                        interval_seq[-1])

            dict_res_update = self.find_rule_binning_interval(pred, var, begin, end, include_exclude)

            if dict_res_update is not None:
                if self.debug_mode:
                    print("dict_res_update is not None", dict_res_update)
                binning_dict[tuple(interval_seq)] = dict_res_update
        return binning_dict

    def _merge_all_intervals_p_var(self):
        ## TODO: based on the binning dict after all the loop is over
        return None

    def _find_existential_num_(self, feat_pd_, var, pred):

        pca_body_size_existential = feat_pd_.drop_duplicates(self.rule.conclusion.atom_raw_variables).shape[0]
        support_existential = self.var_pred_to_relaxed_supp[var][pred]
        conf_existential = support_existential / pca_body_size_existential
        has_existential = True if conf_existential > self.min_conf_pr else False
        return has_existential, pca_body_size_existential, support_existential

    def add_numerical_pred_to_rule(self, var, preds):

        step1 = time()

        step2 = time()

        feat_pd_all, suitable_pred_str = self.create_df_from_query(preds, var)
        for pred, feat_pd_ in feat_pd_all.groupby(suitable_pred_str):
            if self.debug_mode:
                sss = feat_pd_.groupby('label').count()
                sss = sss.reset_index(drop=True)
                ttt = sss.loc[1]['feat'] if 1 in sss.index else 0
                fff = sss.loc[0]['feat'] if 0 in sss.index else 0
                self.imbalanced_perc.append(ttt / (ttt + fff))

            step3 = time()

            # feat_pd_.sort_values(by="feat", inplace=True)  # for visualization
            # self.plot_scatt(fea.size_head_rt_pd_)  # for visualization
            if self.debug_mode:
                self._create_dict_df_pred_var(pred, var, feat_pd_)
            ###TODO: if reached here with pred and var, possible that bining will be done on 2 or more features. keep them and make a call

            for binning_mode in self.binningtechniques:
                # TODO : try adding existial here..
                has_existential, pca_body_size_existential, support_existential = \
                    self._find_existential_num_(feat_pd_, var, pred)

                if has_existential:
                    if self.debug_mode:
                        print(has_existential, 'EEEXISTENTIAL')
                        print(pca_body_size_existential, support_existential)

                    m_e = MeasureExistential(rule=self.rule, graph=self.graph, base_q=self._base_query_df,
                                             pred=pred, var_num=var,
                                             pca_body_size_pre_computed=pca_body_size_existential,
                                             support_pre_computed=support_existential, verbose=False)
                    _dict = m_e.toDict()
                    _dict["interval"] = -2
                    self.dict_bin_new_rules[binning_mode].append(_dict)

                binning = self.start_binning(feat_pd_, binning_mode)
                step4 = time()

                binning_dict = self.find_possible_rules_binning(binning, feat_pd_, pred, var)
                step42 = time()

                if not binning_dict:
                    continue

                # TODO: to have the rules that are consecutive (in terms of interval), take care here, and recalculate their quality measures
                if self.merge_bc_intervals and len(binning_dict) > 1:
                    binning_dict = self._merge_biggest_consecutive_intervals_binning_dict(binning, feat_pd_,
                                                                                          binning_dict)

                ## TODO: instead of merging the biggest consecituve, merge all the intervals
                # if self.merge_all_intervals and len(binning_dict) > 1:
                #    self._merge_all_intervals_p_var()

                # for interval, binning_dict_interval in binning_dict.items():
                #    self.rule_loader.write_numerical_pred_rules(self.rule, binning_dict_interval)
                step45 = time()

                for interval, _dict in binning_dict.items():
                    _dict["interval"] = interval
                    self.dict_bin_new_rules[binning_mode].append(_dict)

                if self.debug_mode:
                    try:
                        print(f'binning_mode = {binning_mode}')
                        print(binning.build_table())
                        self.plot_scatt(feat_pd_, binning.splits)
                    except:
                        print('CANNOT DRAW PROBABLY ALL 0 which means existential')

                    step5 = time()

                    print('step2 - step1:', step2 - step1)
                    print('step3 - step25', step3 - step2)
                    print('step4 - step3', step4 - step3)
                    print('step42 - step4', step42 - step4)
                    print('step45 - step42', step45 - step42)
                    print("\n")

        # for binning_mode in self.binningtechniques: #TODO: what is this for loop for? existential does not need binning

    def find_rule_binning_interval(self, pred, var, begin, end, new_supp, new_pca_bdsize, include_exclude):

        quality_measure_new_rule = MeasureEnriched(rule=self.rule, graph=self.graph, base_q=self._base_query_df,
                                                   pred=pred, var_num=var,
                                                   begin_interval=begin, end_interval=end,
                                                   include_exclude=include_exclude,
                                                   support_pre_computed=new_supp,
                                                   pca_body_size_pre_computed=new_pca_bdsize,
                                                   verbose=False)

        if not self._checker_new_conf_better_than_parent_confidence(quality_measure_new_rule.pca_confidence):
            if self.debug_mode:
                print('DID NOT PASS quality_measure_new_rule.pca_confidence', quality_measure_new_rule.pca_confidence)
            return None

        if self.debug_mode:
            print('EVERYTHING OK PASS NEW RULE pca_confidence OK!!!!')
        # new rule
        return quality_measure_new_rule.toDict()

    def find_possible_rules_binning(self, binning, df, pred, var):
        binning_dict = {}
        if self.debug_mode:
            print(f'considering pred: {pred}, var: {var}')
        sorted_event_rate = binning.event_rate
        # TODO: can potentially remove sorting the bins now, cause no earlier pruning can be done
        for _indx, event_rate in enumerate(sorted_event_rate):
            if self.debug_mode:
                print('_indx, event_rate', _indx, event_rate)
            passed, include_exclude = self._checker_prune_better_than_parent_confidence(event_rate)
            if not passed:  # TODO: not sure if this rule causes pruning when should not
                if self.debug_mode:
                    print("did not pass the thr for parent conf...")
                continue

            interval_indx = binning.sort_indx[_indx]
            begin, end = binning.bins_intervals[_indx][0], binning.bins_intervals[_indx][1]

            new_supp = self.compute_new_supp_from_df(df, begin, end, include_exclude)

            new_pca_bdsize = self.compute_new_pca_bdysize_from_df(df, begin, end, include_exclude)

            if not self._checker_satisfy_minhc(new_supp):
                if self.debug_mode:
                    print("did not pass the thr for minhc= continue")
                continue
            if self.debug_mode:
                print("include_exclude", include_exclude)
                print("MINHC PASSED OK!!!! supp: ", new_supp)

            dict_res = self.find_rule_binning_interval(pred, var, begin, end, new_supp, new_pca_bdsize, include_exclude)

            if dict_res is not None:
                if self.debug_mode:
                    print("new_pca with df ", new_pca_bdsize)
                    print("with query :", dict_res["pcaBodySize"])
                binning_dict[interval_indx] = dict_res

            if len(binning_dict) == 10:
                if self.debug_mode:
                    print("pruning here.. too may intervals considered")
                    # keep a counter to see
                break

        if self.debug_mode:
            if len(binning_dict) > 0:
                print('Done. returning  binning_dict ok!!!!')

        return binning_dict

    def _checker_prune_better_than_parent_confidence(self, event_rate):
        # event_rate = opt.n_event[interval_indx]/ (opt.Non-event[interval_indx]+ opt.n_event[interval_indx])
        # is this really correct ? to do pruning using the event rate given that maximum reduction from
        # support is opt.n_event and maximum reduction from denom is opt.n_event+opt.Non-event --> gand nazanim bikhod prune
        # konim?

        if 1 - event_rate > self.min_conf_pr:
            return True, 'exclude'

        if event_rate > self.min_conf_pr:
            return True, 'include'

        return False, ''

    def _checker_new_conf_better_than_parent_confidence(self, new_pca_conf):
        # return True if new_pca_conf > self.rule.pcaConfidence else False
        return True if new_pca_conf > self.min_conf_pr else False

    def compute_new_supp_from_df(self, df, begin, end, include_exclude, by="feat", grp_by="label"):
        df_tmp = df[df[grp_by] == 1]
        if include_exclude == 'exclude':
            df_t = df_tmp[~df[by].between(begin, end, inclusive="left")]
        elif include_exclude == 'include':
            df_t = df_tmp[df[by].between(begin, end)]
        else:
            raise Exception(f'bug, {include_exclude} should be exclude or include')

        new_supp = df_t.drop_duplicates(subset=self.rule.conclusion.atom_raw_variables).shape[0]

        return new_supp

    def compute_new_pca_bdysize_from_df(self, df, begin, end, include_exclude):

        if include_exclude == 'exclude':
            df_t = df[~df["feat"].between(begin, end, inclusive="left")]
        elif include_exclude == 'include':
            df_t = df[df["feat"].between(begin, end)]
        else:
            raise Exception(f'bug, {include_exclude} should be exclude or include')

        new_pca = df_t.drop_duplicates(subset=self.rule.conclusion.atom_raw_variables).shape[0]
        return new_pca

    def create_df_from_query(self, preds, var):
        # TODO: there might be a bug here, make sure 0/1 label is based on the functionality
        # ## get the functionality based on the parent rule and define the labels accordingly

        q_out_str, suitable_pred_str = query_rule_add_df(self._base_query_df, self.rule.rule_variables, preds, var)
        df = self.graph.query_dataframe(q_out_str)

        # TODO: there might be a bug here. Do we check weather all variables or not
        # PlaceOfWork(X,Y) and placeOfDeath -> PlaceOfBirth(X,Y)

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

        df = df.rename(columns={'anythinglit': 'feat'})
        df['feat'] = df['feat'].astype(float)
        return df, suitable_pred_str

    def plot_scatt(self, feat_pd_, splits=[], by='feat', grp_by='label'):
        scatter_x = np.array(feat_pd_[by])
        scatter_y = np.array(feat_pd_[by])
        group = np.array(feat_pd_[grp_by])
        cdict = {0: 'red', 1: 'blue'}
        fig, ax = plt.subplots()
        for g in np.unique(group):
            ix = np.where(group == g)
            if g == 0:
                ss = 100
            else:
                ss = 20
            ax.scatter(scatter_x[ix], scatter_y[ix], c=cdict[g], label=g, s=ss)
        ax.legend()
        for s in splits:
            plt.axvline(x=s)
        plt.show()

    def start_binning(self, df_clf, binning_mode):
        binning = None
        if binning_mode == "opt":
            binning = OptimalBinningFitStats(df_clf)
        elif binning_mode == 'custom_mdlp':
            binning = CustomMDLPFitStats(df_clf)
        elif binning_mode == "quantile":
            binning = UnsupervisedBinning(df_clf, prebinning_method="quantile")
        elif binning_mode == "uniform":
            binning = UnsupervisedBinning(df_clf, prebinning_method="uniform")
        elif binning_mode == "mdlp":
            binning = MDLPFitStats(df_clf)
        elif binning_mode == "tree":
            binning = TreeBinnig(df_clf)
        else:
            NotImplementedError()
        return binning
