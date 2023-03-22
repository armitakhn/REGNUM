from dataclasses import dataclass, field
import pandas as pd
from custom_exception import NotTriplesError, NotLabelError
import tqdm
from urllib.error import HTTPError
from urllib import parse
from urllib.request import urlopen


@dataclass
class BaseDataLoader:
    """
    path_t: path to the train data
    path_numerical_preds: path to numerical predicates
    _numerical_preds: if none, to
    be found using range - schema , else: path to numerical preds
    """
    path_t: str = None
    path_numerical_preds: str = None
    delim: str = "\t"
    numerical_preds: list = field(default_factory=list)
    prefix: str = "http://"

    def __post_init__(self):
        self.read_data(self.path_t)
        self._get_num_preds()
        self._is_num_idx_finder()

    def read_data(self, path):
        self.df = pd.read_csv(path, sep=self.delim, index_col=False, names=["subject", "predicate", "object"])
        self.df = self.df.drop_duplicates().reset_index(drop=True)

        if self.df.shape[1] != 3:
            raise NotTriplesError("The given input file does not correspond to a proper rdf KG... check input.")

    def _get_num_preds(self):
        if self.path_numerical_preds is None:
            # TODO, look into self.df for all predicates that have range only numbers
            raise NotImplementedError

        else:
            try:
                f = open(self.path_numerical_preds, "r")
                self.numerical_preds = []
                for line in f.readlines():
                    self.numerical_preds.append(line.strip())

            except FileNotFoundError:
                f = urlopen(self.path_numerical_preds)
                self.numerical_preds = []
                for line in f.readlines():
                    self.numerical_preds.append(line.decode().strip())

    def _is_num_idx_finder(self):
        self._is_num = self.df.predicate.isin(self.numerical_preds)

    def add_prefix_url_quote(self):

        def prepare_data_url(x):
            if not x.startswith(self.prefix):
                return f'{self.prefix}{parse.quote(str(x))}'
            else:
                return f'{x[:len(self.prefix)]}{parse.quote(str(x[len(self.prefix):]))}'

        self.df.loc[self._is_num, ["subject", "predicate"]] = self.df.loc[
            self._is_num, ["subject", "predicate"]].applymap(prepare_data_url)

        self.df.loc[~self._is_num] = self.df.loc[~self._is_num].applymap(prepare_data_url)
        self.numerical_preds = list(self.df[self._is_num].predicate.unique())

    def _split_dfs_num_not_num(self):
        self.df_num = self.df[self._is_num]
        self.df_not_num = self.df[~self._is_num]


@dataclass
class FBDBCustomDataLoader(BaseDataLoader):
    path_entities_label: str = None
    replace_triples_with_label: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.replace_triples_with_label:
            self._replace_id_with_label()

        self._split_dfs_num_not_num()
        self._clean_nums()
        self.set_numerical_preds()

    def _replace_id_with_label(self):
        def _replace_via_merge(df):
            df = df.merge(df_label_ent, left_on='subject', right_on='entity', how='left')
            df = df.merge(df_label_ent, left_on='object', right_on='entity', how='left')

            df.loc[df['label_x'].isnull(), 'label_x'] = df['subject']
            df.loc[df['label_y'].isnull(), 'label_y'] = df['object']
            df = df[['label_x', 'predicate', 'label_y']]

            df = df.rename(
                columns={'label_x': 'subject', 'predicate': 'predicate', 'label_y': 'object'})
            return df

        df_label_ent = pd.read_csv(self.path_entities_label, sep="\t", index_col=False, names=["entity", "label"])

        # dict_ent_2_label = dict(zip(df_label_ent.entity, df_label_ent.label))

        if df_label_ent.shape[1] != 2:
            raise NotLabelError("The given input files do not have two columns.. check ent props to label files")

        self.df = _replace_via_merge(self.df)
        self.df = self.df.replace('<http://', '', regex=True).replace('>', '', regex=True).replace(' ', '_', regex=True)

    def _clean_nums(self):
        self.df.object[self._is_num] = self.df.object[self._is_num].apply(float)

    def set_numerical_preds(self):
        self.numerical_preds = list(self.df_num.predicate.unique())


@dataclass
class WikiDataDataLoader(BaseDataLoader
                         ):
    path_predicates_label: str = None
    path_entities_label: str = None
    path_entities_type: str = None
    replace_triples_with_label: bool = False

    def __post_init__(self):

        super().__post_init__()
        self._split_dfs_num_not_num()
        self._read_ent_types()
        if self.replace_triples_with_label:
            self._replace_id_with_label()
            rep_dict_pred: dict = self._dict_numerical_preds_wikidata_to_label()
            self._replace_pred_numerical_dict(rep_dict_pred)
        self._clean_numerical()
        self._concat_num_not_num_df()
        self.set_numerical_preds()

    def _read_ent_types(self):
        if self.path_entities_type is None:
            self.df_ent_type = None
        else:
            self.df_ent_type = pd.read_csv(self.path_entities_type, sep="\t", index_col=False, names=["ent", "type"])

    def _replace_id_with_label(self):
        def _replace_via_merge(df):
            df = df.merge(df_label_ent, left_on='subject', right_on='entity', how='left')
            df = df.merge(df_label_pred, on='predicate', how='left')
            df = df.merge(df_label_ent, left_on='object', right_on='entity', how='left')
            df.loc[df['label_x'].isnull(), 'label_x'] = df['subject']
            df.loc[df['label_y'].isnull(), 'label_y'] = df['predicate']
            df.loc[df['label'].isnull(), 'label'] = df['object']
            df = df[['label_x', 'label_y', 'label']]

            df = df.rename(
                columns={'label_x': 'subject', 'label_y': 'predicate', 'label': 'object'})
            return df

        def _replace_ent_ids_with_labels(df):
            df = df.merge(df_label_ent, left_on='ent', right_on='entity', how='left')
            df.loc[df['entity'].isnull(), 'entity'] = df['ent']
            df = df[['label', 'type']]
            df = df.rename(columns={'label': 'ent'})
            return df

        df_label_pred = pd.read_csv(self.path_predicates_label, sep="\t", index_col=False, names=["predicate", "label"])
        df_label_ent = pd.read_csv(self.path_entities_label, sep="\t", index_col=False, names=["entity", "label"])

        if df_label_pred.shape[1] != 2 or df_label_ent.shape[1] != 2:
            raise NotLabelError("The given input files do not have two columns.. check ent props to label files")

        self.df_not_num = _replace_via_merge(self.df_not_num)
        self.df_not_num = self.df_not_num.replace(' ', '_', regex=True)

        self.df_num = _replace_via_merge(self.df_num)
        # self.df_num = self.df_num.replace(' ', '_', regex=True)
        # TODO : P625_Longtiude the numericals don't have labels as is, separate and replace DONE

        self.df_ent_type = _replace_ent_ids_with_labels(self.df_ent_type)

    def _clean_numerical(self):
        def _clean_type_date(row):
            parts = row.split("^^")
            if parts[1][:-1].split("#")[-1] == "dateTime":
                return int(parts[0].split("T")[0].replace("-", ""))
            else:
                return float(parts[0])

        self.df_num.object = self.df_num.object.apply(_clean_type_date)

    def _dict_numerical_preds_wikidata_to_label(self):
        from data_preprocessing import get_label_name
        self.df_num.predicate.unique()

        replace_preds_num_dict = {}
        for pred in tqdm.tqdm(self.df_num.predicate.unique()):
            l_pred = pred.split("_")
            if len(l_pred) == 1:
                replace_preds_num_dict[pred] = pred
            else:
                n = 0
                success = False
                try_time = 40
                while n < try_time:
                    try:
                        s = "_".join(get_label_name(pred) for pred in l_pred)
                        replace_preds_num_dict[pred] = s
                        success = True
                        break
                    except HTTPError:
                        n += 1
                        if n > 30:
                            print(pred)
                            print(n)
                        pass

                if not success:
                    raise Exception(f'failed to fetch data for {l_pred} after {try_time} times')

        # self.numerical_preds = list(replace_preds_num_dict.keys())
        return replace_preds_num_dict

    def _replace_pred_numerical_dict(self, replace_preds_num_dict):
        self.df_num.predicate = self.df_num.predicate.apply(lambda x: replace_preds_num_dict[x])
        self.df_num = self.df_num.replace(' ', '_', regex=True)
        # print(self.df_num)

    def _concat_num_not_num_df(self):
        self.df = pd.concat([self.df_not_num, self.df_num])
        self.df = self.df.drop_duplicates(keep='first')

    def set_numerical_preds(self):
        self.numerical_preds = list(self.df_num.predicate.unique())


@dataclass
class GeneralDataLoader(BaseDataLoader):

    def __post_init__(self):
        super().__post_init__()
        # TODO - boolean input for tabular or rdf data input

        self.add_prefix_url_quote()  # TODO: if the input does not have prefix of its own should be added

        self.df.object[self._is_num] = self.df.object[self._is_num].apply(float)
        self.set_numerical_preds()

    def set_numerical_preds(self):
        self.numerical_preds = list(self.df[self._is_num].predicate.unique())