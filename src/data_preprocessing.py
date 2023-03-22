from data_loader import WikiDataDataLoader
from dataclasses import dataclass
from custom_exception import NotTypeError
from SPARQLWrapper import SPARQLWrapper, JSON
import conf
import pandas as pd


def get_label_name(inp):
    sparql = SPARQLWrapper(conf.ENDPOINTSPARQL)
    type_label = None
    query = f"""
    select ?label
    where {{
            wd:{inp} rdfs:label ?label .
      FILTER (langMatches( lang(?label), "EN" ) )
          }}
    LIMIT 1
    """
    # TODO: query only works for the Wikidata endpoint..
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        type_label = result["label"]["value"]
        break

    if type_label is None:
        type_label = inp

    return type_label


@dataclass
class TypeDataClassifier:
    dataloader: WikiDataDataLoader
    typeClassify: str
    replaceTLSPARQL: bool = False

    def __post_init__(self):
        if self.replaceTLSPARQL:
            self.type_label = get_label_name(self.typeClassify)
        self.res_sub_types = self._get_all_subclasses_in_types()
        self._constrain_to_type()
        self._add_is_type_preds()

    def _get_all_subclasses_in_types(self):
        all_types = self.dataloader.df_ent_type.type.unique()
        res_sub_types = []
        sparql = SPARQLWrapper(conf.ENDPOINTSPARQL)
        query = f"""
        select ?subClass
        where{{?subClass wdt:P279* wd:{self.typeClassify} .}}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # c, d = 0 ,0
        for result in results["results"]["bindings"]:
            # c += 1
            subclass_t = result["subClass"]["value"]
            subclass_t = subclass_t.split("/")[-1]
            if subclass_t in all_types:
                res_sub_types.append(subclass_t)
            else:
                # d += 1
                continue
        return res_sub_types

    def _constrain_to_type(self):
        if self.dataloader.df_ent_type is not None:
            df_types = self.dataloader.df_ent_type
            self.df_type = df_types[df_types.type.isin(self.res_sub_types)]

        else:
            raise NotTypeError("The file for specifiying ent to types has not be provided..")
            # TODO: if known KGs, can also retrieve corresponding types from their endpoints

    def _add_is_type_preds(self):
        df_m = self.dataloader.df_not_num.copy()
        subject_of_typeClassify = set(df_m.subject.unique()).intersection(set(self.df_type.ent.unique()))
        ## TODO: make sure no bug here
        subj = list(subject_of_typeClassify)
        pred = [f"is_{self.type_label}"] * len(subj)
        obj = ["True"] * len(subj)
        df_is_type = pd.DataFrame.from_dict({"subject": subj, "predicate": pred, "object": obj})

        subj_n = list(set(df_m.subject.unique()) - subject_of_typeClassify)
        pred_n = [f"is_{self.type_label}"] * len(subj_n)
        obj_n = ["False"] * len(subj_n)
        df_is_not_type = pd.DataFrame.from_dict({"subject": subj_n, "predicate": pred_n, "object": obj_n})

        self.df_not_num_type = pd.concat([df_m, df_is_type, df_is_not_type])

        # assert self.pd_df_not_num_type.shape[0] == df_m.shape[0] + df_is_type.shape[0]


@dataclass
class PredInstanceDataClassifier:
    dataloader: WikiDataDataLoader
    predClassify: str
    instClassify: str

    ## TODO :  if needed, user can specify if the instantiation is based on object or subject. For now only object is supported.

    def __post_init__(self):
        df_true, df_false = self._find_pred_inst()
        self._add_pred_inst_class(df_true, df_false)

    def _find_pred_inst(self):
        # _new_pred = f"{self.predClassify} {self.instClassify}"
        _new_pred = "usa"
        df = self.dataloader.df_not_num.copy()
        pred_idx = df.predicate == self.predClassify
        df = df[pred_idx]
        df.predicate = _new_pred

        obj_idx_true = df.object == self.instClassify
        df_true, df_false = df[obj_idx_true], df[~obj_idx_true]
        df_true.object = "True"
        df_false.object = "False"
        return df_true, df_false

    def _add_pred_inst_class(self, df_true, df_false):
        self.df_not_num_pred_inst = pd.concat([self.dataloader.df_not_num, df_true, df_false])
