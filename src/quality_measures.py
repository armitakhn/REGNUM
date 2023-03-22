from dataclasses import dataclass
from rdflib.graph import Graph
from rule import Rule
from graph_query import query_support, query_body_size, query_head_size, query_pca_body_size, \
    query_pca_body_size_num_pred_interval, query_body_size_num_pred_interval, query_count_groundings
import random
from tqdm import tqdm
from graph_data import RDFGraph
import numpy as np


@dataclass
class BaseNumMeasure:
    rule: Rule
    graph: RDFGraph
    support_pre_computed: int
    pca_body_size_pre_computed: int

    pred: str  # population
    var_num: str  #
    base_q: str

    def __post_init__(self):
        self.support = self.support_pre_computed
        self.pca_body_size = self.pca_body_size_pre_computed

        self.body_size = self.compute_body_size_num()
        self.standard_confidence = self.compute_standard_confidence()
        self.pca_confidence = self.compute_pca_confidence()
        self.head_coverage = self.support / self.rule.size_head_r

    def compute_body_size_num(self):
        qbsn = query_body_size_num_pred_interval(self.rule, self.pred, self.var_num, self.begin_interval,
                                                 self.end_interval, self.include_exclude)
        if self.verbose:
            print("compute body size numerical \n")
        return self.graph.query_count(qbsn)

    def compute_pca_confidence(self):
        if self.pca_body_size < 1:
            return 0
        return self.support / self.pca_body_size

    def compute_standard_confidence(self):
        if self.body_size < 1:
            return 0
        return self.support / self.body_size

    @property
    def compute_f_score(self):
        return 2 * (self.pca_confidence * self.head_coverage) / (self.pca_confidence + self.head_coverage)

    def toDict(self):

        return {"parent_rule": self.rule,
                "support": self.support,
                "pcaBodySize": self.pca_body_size,
                "bodySize": self.body_size,
                'f_score': self.compute_f_score,
                "pcaConfidence": self.pca_confidence,
                "stdConfidence": self.standard_confidence,
                "headCoverage": self.head_coverage,
                "pred": self.pred,
                "var_num": self.var_num,
                "beginInterval": self.begin_interval,
                "endInterval": self.end_interval,
                "functionalVariable": self.rule.functionalVariable,
                "include_exclude": self.include_exclude,
                "conclusion": self.rule.conclusion,
                "size_hypothese": self.rule.size_hypotheses + 1,
                "enriched_rule": self.set_hypotheses()}  # TODO: if more than 1 num pred , fix this


@dataclass
class MeasureExistential(BaseNumMeasure):
    pca_body_size_pre_computed: int
    verbose: bool = False

    def __post_init__(self):
        self.include_exclude: str = 'include'  # TODO: change this to existential
        self.begin_interval: float = -np.inf
        self.end_interval: float = np.inf
        self.pca_body_size = self.pca_body_size_pre_computed
        super().__post_init__()

    def set_hypotheses(self):
        # TODO: change this later according to the definition that will be given in the paper
        # hypo = f' ∃ ?x {self.var_num} {self.pred} ?x {self.rule.rule}'
        hypo = f'{self.var_num} {self.pred} ?any  {self.rule.rule}'

        return hypo


@dataclass
class MeasureEnriched(BaseNumMeasure):
    # ?a, ?b
    begin_interval: float
    end_interval: float
    include_exclude: str
    verbose: bool = False

    def __post_init__(self):
        super().__post_init__()

    def compute_pca_body_size_num(self):
        # TODO:check with df without query
        qpcabsn = query_pca_body_size_num_pred_interval(self.rule, self.base_q, self.pred, self.var_num,
                                                        self.begin_interval,
                                                        self.end_interval, self.include_exclude)
        q = self.graph.query_count(qpcabsn)

        if self.verbose:
            print("compute pca body size numerical \n")

        return q

    def check_pca_bdy_size_df_query(self):
        pca_bdy_query = self.compute_pca_body_size_num()
        if self.pca_body_size == pca_bdy_query:
            return True
        else:
            return False

    def set_hypotheses(self):
        # TODO: change this later according to the definition that will be given in the paper
        _term = 'NOT BETWEEN' if self.include_exclude == 'exclude' else 'BETWEEN'
        hypo = f'{self.var_num} {self.pred} ?any ?any {_term} [{self.begin_interval}, {self.end_interval}] {self.rule.rule}'
        return hypo


@dataclass
class RuleMeasure:
    rule: Rule
    graph: RDFGraph
    verbose: bool = False

    def __post_init__(self):

        self.func_var = self.compute_functionality()
        self.rule.set_functionalVariable(self.func_var)  # needed for pca_body_size
        self.head_size = self.compute_head_size()
        self.support = self.compute_support()
        self.body_size = self.compute_body_size()
        self.head_coverage = self.compute_head_coverage()
        self.pca_body_size = self.compute_pca_body_size()
        self.standard_confidence = self.compute_standard_confidence()
        self.pca_confidence = self.compute_pca_confidence()

    def compute_functionality(self):
        qfu_subj = query_count_groundings(self.rule, on='subject')
        subj_count = self.graph.query_count(qfu_subj)

        qfu_obj = query_count_groundings(self.rule, on='object')
        obj_count = self.graph.query_count(qfu_obj)
        return self.rule.conclusion.subject if subj_count > obj_count else self.rule.conclusion.objectD

    def compute_pca_confidence(self):
        if self.verbose:
            print("pca confidence: ", self.support / self.pca_body_size)
            print("support", self.support)
            print("pca bds", self.pca_body_size)
        return self.support / self.pca_body_size if self.pca_body_size > 0 else 0

    def compute_standard_confidence(self):
        return self.support / self.body_size if self.body_size > 0 else 0

    def compute_support(self):
        qs = query_support(self.rule)
        if self.verbose:
            print("compute support \n")
            print(qs)
        return self.graph.query_count(qs)

    def compute_head_size(self):
        hcq = query_head_size(self.rule)
        return self.graph.query_count(hcq)

    def compute_head_coverage(self):
        hcq = query_head_size(self.rule)
        self.head_size = self.graph.query_count(hcq)
        if self.verbose:
            print("compute head coverage \n")
            print(self.support / self.head_size)
        return self.support / self.head_size

    def compute_body_size(self):
        qbs = query_body_size(self.rule)
        if self.verbose:
            print("compute body size \n")
            print(qbs)
        return self.graph.query_count(qbs)

    def compute_pca_body_size(self):
        qpcabs = query_pca_body_size(self.rule)
        if self.verbose:
            print("compute pca body size \n")
            print(qpcabs)
        return self.graph.query_count(qpcabs)


def sanityCheckMeauresAmie(rule: Rule, graph: Graph):
    ms = RuleMeasure(rule, graph)
    supp, bd, pcabd = 0, 0, 0
    try:
        assert ms.support == rule.support
    except AssertionError:
        print(f"Correct support {rule.support}")
        print(f"calculated support {ms.support}")
        supp += 1
    try:
        assert ms.body_size == rule.bodySize
    except AssertionError:
        print(f"Correct bodySize {rule.bodySize}")
        print(f"calculated bodySize {ms.body_size}")
        bd += 1
    try:
        assert abs(ms.head_coverage - rule.headCoverage) < 1e-9
    except AssertionError:
        print(f"Correct headCoverage {rule.headCoverage}")
        print(f"calculated headCoverage {ms.head_coverage}")
    try:
        assert abs(ms.standard_confidence - rule.stdConfidence) < 1e-9
    except AssertionError:
        print(f"Correct stdConfidence {rule.stdConfidence}")
        print(f"calculated stdConfidence {ms.standard_confidence}")
    try:
        assert ms.pca_body_size == rule.pcaBodySize
    except AssertionError:
        print(f"Correct pcaBodySize {rule.pcaBodySize}")
        print(f"calculated pcaBodySize {ms.pca_body_size}")
        pcabd += 1

    try:
        assert abs(ms.pca_confidence - rule.pcaConfidence) < 1e-9
    except AssertionError:
        print(f"Correct pca_confidence {rule.pcaConfidence}")
        print(f"calculated pca_confidence {ms.pca_confidence}")

    return supp, bd, pcabd


def sanitycheckRandom(rules, graph: Graph):
    supps, bds, pcabds = 0, 0, 0
    rules_sample = random.sample(list(range(len(rules))), len(rules))
    for rule_id in rules_sample:
        supp, bd, pcabd = sanityCheckMeauresAmie(rules[rule_id], graph)
        supps += supp
        bds += bd
        pcabds += pcabd

    print(supps, bds, pcabds)
