from optbinning import OptimalBinning, MDLP
from optbinning.binning.prebinning import PreBinning

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


@dataclass
class BaseBinning:
    df: pd.DataFrame
    feat_name: str = "feat"
    label_name: str = "label"

    def __post_init__(self):
        self.event_rate = self._n_event / (self._n_nonevent + self._n_event)
        self.sort_indx = self.event_rate.argsort()[::-1]

        self.n_nonevent = self._n_nonevent[self.sort_indx]  # sort by the index that event rate has been sorted
        self.n_event = self._n_event[self.sort_indx]

        self.n_records = self.n_event + self.n_nonevent
        self.event_rate = self.n_event / (self.n_nonevent + self.n_event)
        self.t_n_nonevent = self.n_nonevent.sum()
        self.t_n_event = self.n_event.sum()
        self.t_n_records = self.t_n_nonevent + self.t_n_event
        self.t_event_rate = self.t_n_event / self.t_n_records

        self.p_records = self.n_records / self.t_n_records  # count(%)
        # self.p_event = self.n_event / self.t_n_event
        # self.p_nonevent = self.n_nonevent / self.t_n_nonevent

        self._bins_intervals()
        self.bins_intervals = self.bins_intervals[self.sort_indx]

    def _bins_intervals(self):
        bins = np.concatenate([[-np.inf], self.splits, [np.inf]])
        self.bins_intervals = []
        for i in range(len(bins) - 1):
            if np.isinf(bins[i]):
                b = (bins[i], bins[i + 1])
            else:
                b = (bins[i], bins[i + 1])

            self.bins_intervals.append(b)

        self.bins_intervals = np.array(self.bins_intervals)  # [self.sort_indx]

    def recompute_stats_new_interval(self, begin_indx, end_indx):
        indices_k = list(range(begin_indx, end_indx + 1))
        n_event_int = self.n_event[indices_k].sum()
        n_nonevent_int = self.n_nonevent[indices_k].sum()
        event_rate_int = n_event_int / (n_nonevent_int + n_event_int)
        return n_event_int, n_nonevent_int, event_rate_int

    def target_info_bins(self):
        def _map_feat_to_bin(row):
            for i, split in enumerate(self.splits):
                if row <= split:
                    return i
            return len(self.splits)

        self.df["mapping"] = self.df[self.feat_name].apply(_map_feat_to_bin)
        _gbdf = self.df.groupby(["mapping", self.label_name]).size().unstack(fill_value=0)
        if 0 in _gbdf:
            _n_nonevent = _gbdf[0].values
        else:
            _n_nonevent = np.array([0] * _gbdf.shape[0])

        if 1 in _gbdf:
            _n_event = _gbdf[1].values
        else:
            _n_event = np.array([1] * _gbdf.shape[0])

        return _n_event, _n_nonevent

    def build_table(self):
        bin_int = [f"{bi[0], bi[1]}" for bi in self.bins_intervals]
        df_table = pd.DataFrame({
            "intervals": bin_int,
            "Count": self.n_records,
            "Count (%)": self.p_records,
            "Non-event": self.n_nonevent,
            "Event": self.n_event,
            "Event rate": self.event_rate,

        })
        return df_table



@dataclass
class UnsupervisedBinning(BaseBinning):
    min_bin_size: float = 0.001
    prebinning_method: str = 'uniform'

    def __post_init__(self):
        self.opt = PreBinning(problem_type="classification", n_bins=4,
                          min_bin_size=self.min_bin_size, method=self.prebinning_method)

        self.opt.fit(self.df[self.feat_name].values, self.df[self.label_name])
        self.splits = self.opt.splits

        self._n_event, self._n_nonevent = self.target_info_bins()

        super().__post_init__()


@dataclass
class OptimalBinningFitStats(BaseBinning):
    solver: str = "mip"
    min_bin_size: float = 0.001
    prebinning_method: str = 'mdlp'

    def __post_init__(self):
        self.opt = OptimalBinning(name=self.feat_name, dtype="numerical", solver=self.solver,
                                min_bin_size=self.min_bin_size, prebinning_method=self.prebinning_method)

        self.opt.fit(self.df[self.feat_name], self.df[self.label_name])
        self.splits = self.opt.splits

        self._n_event = self.opt._n_event[:-2]
        self._n_nonevent = self.opt._n_nonevent[:-2]

        super().__post_init__()

    def build_table(self) -> pd.DataFrame:
        binning_table = self.opt.binning_table
        return binning_table.build()



@dataclass
class CustomMDLPFitStats(BaseBinning):

    def __post_init__(self):
        self.mdlp = MDLP(min_samples_split=2, min_samples_leaf=1, max_candidates=36)
        self.mdlp.fit(self.df[self.feat_name], self.df[self.label_name])

        self.splits = self.mdlp.splits
        print(self.splits)
        print('olalaaaa')
        self._n_event, self._n_nonevent = self.target_info_bins()
        super().__post_init__()



@dataclass
class MDLPFitStats(BaseBinning):

    def __post_init__(self):
        self.mdlp = MDLP(min_samples_split=2, min_samples_leaf=1, max_candidates=36)
        self.mdlp.fit(self.df[self.feat_name], self.df[self.label_name])

        self.splits = self.mdlp.splits
        self._n_event, self._n_nonevent = self.target_info_bins()
        super().__post_init__()


@dataclass
class TreeBinnig(BaseBinning):
    min_samples_leaf: int = 2
    max_depth: int = 4
    criterion: str = 'entropy'

    def __post_init__(self):
        self.tree_model = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth,
                                                 criterion=self.criterion)
        self.tree_model.fit(self.df[self.feat_name].to_frame(), self.df[self.label_name])
        self.splits = self.find_splits()
        self._n_event, self._n_nonevent = self.target_info_bins()
        super().__post_init__()

    def find_splits(self):
        thr = list(self.tree_model.tree_.threshold)
        thr.reverse()

        splits = []
        try:
            for i in range(len(thr)):
                if thr[i] == -2:
                    if thr[i + 1] != -2:
                        splits.append(thr[i + 1])

            splits.sort()
            return splits
        except IndexError:
            return []
