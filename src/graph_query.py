from rule import Rule
from graph_data import RDFLibGraph
import numpy as np


def query_rule_num_pred(rule, var_preds_dict):
    # var_preds_dict = {'a':[wwm,zz], 'b':[bb,tt]}
    # var = ?a
    # rule = ?a  place of birth  ?b   => ?a  place of death  ?b
    # pred: patronage_1
    # query:
    # SELECT DISTINCT ?a WHERE {
    # ?a <http://place_of_birth> ?b .
    # ?a <http://place_of_death> ?b .
    # ?a <http://patronage_1> ?anythinglit .
    # }

    # TODO check if we should add DISTINCT COUNT

    query = f""" SELECT ( COUNT( DISTINCT * ) AS ?count ) WHERE {{\n"""
    query += f"""SELECT DISTINCT {rule.conclusion.subject} {rule.conclusion.objectD} WHERE {{"""
    for hyp in rule.hypotheses:
        query += f'\n{hyp.atom} .'

    query += f'\n{rule.conclusion.atom} .'
    enum = 0
    for var, num_preds in var_preds_dict.items():
        for num_pred in num_preds:
            query += f'\n{var} <{num_pred}> ?anythinglit_{enum} .'
            enum += 1
    return query + """\n}}"""


def query_select_useful_preds(rule, var):
    query = f""" SELECT DISTINCT ?useful_preds  WHERE {{\n"""
    query += f"""SELECT DISTINCT {rule.conclusion.subject} ?useful_preds {rule.conclusion.objectD} WHERE {{"""
    for hyp in rule.hypotheses:
        query += f'\n{hyp.atom} .'
    query += f'\n{rule.conclusion.atom} .'
    query += f'\n{var} ?useful_preds ?anythinglit .'
    return query + """\n}}"""


'''
def query_rule_num_pred(rule, num_pred, var):
    # var = ?a
    # rule = ?a  place of birth  ?b   => ?a  place of death  ?b
    # pred: patronage_1
    # query:
    # SELECT DISTINCT ?a WHERE {
    # ?a <http://place_of_birth> ?b .
    # ?a <http://place_of_death> ?b .
    # ?a <http://patronage_1> ?anythinglit .
    # }

    # TODO check if we should add DISTINCT COUNT

    query = f"""SELECT (COUNT (DISTINCT {var}) AS ?count) WHERE {{"""
    for hyp in rule.hypotheses:
        query += f'\n{hyp.atom} .'
    query += f'\n{rule.conclusion.atom} .'
    query += f'\n{var} <{num_pred}> ?anythinglit .'
    return query + """\n}"""
'''


def query_rule_add_base(rule, even_not_func_enrich_g: bool):
    # rule = ?a  place of birth  ?b   => ?a  place of death  ?b
    # SELECT
    # DISTINCT * WHERE
    # {
    # ?a < http: // place_of_birth > ?b.
    # ?a < http: // place_of_death > ?anything.

    # TODO: check if distinct correct     query = f"""SELECT DISTINCT ?anythinglit WHERE {{"""
    rep_var = rule.conclusion.objectD
    query = ''
    for hyp in rule.hypotheses:
        query += f'\n{hyp.atom} .'
    if not even_not_func_enrich_g:
        if rule.functionalVariable == rule.conclusion.subject:
            query += f'\n{rule.conclusion.subject} {rule.conclusion.predicate} ?anything  .'
        elif rule.functionalVariable == rule.conclusion.objectD:
            query += f'\n?anything {rule.conclusion.predicate} {rule.conclusion.objectD} .'
            rep_var = rule.conclusion.subject
    else:  # even_not_func_enrich_g == True
        if rule.conclusion.subject in rule.rule_variables:
            query += f'\n?anything {rule.conclusion.predicate} {rule.conclusion.objectD} .'
            rep_var = rule.conclusion.subject
        elif rule.conclusion.objectD in rule.rule_variables:
            query += f'\n{rule.conclusion.subject} {rule.conclusion.predicate} ?anything  .'

    return query, rep_var


def query_main_df(base_query, var_preds_dict):
    query = 'SELECT * WHERE {'
    query += base_query
    iterr = 0
    dict_feature_var_pred = {}
    for var, preds in var_preds_dict.items():
        for pred in preds:
            query += f'\n{var} <{pred}> ?anythinglit{iterr} .'
            dict_feature_var_pred[f"anythinglit{iterr}"] = (var, pred)

            iterr += 1
    query += '\n}'
    return query, dict_feature_var_pred


def query_rule_add_df(base_query, rule_variables, preds, var):
    preds = [f'<{pred}>' for pred in preds]
    query = f"""SELECT {' '.join(rule_variables)} ?suitable_preds ?anything ?anythinglit WHERE {{"""
    query += base_query + f'\n{var} ?suitable_preds ?anythinglit .'
    query += f"\nFILTER(?suitable_preds IN ({', '.join(preds)})) ."

    return query + """\n}""", 'suitable_preds'


def query_count_groundings(rule, on='subject'):
    if on == 'subject':
        query = f"""SELECT (COUNT(DISTINCT {rule.conclusion.subject}) AS ?count) WHERE {{\n"""
    elif on == 'object':
        query = f"""SELECT (COUNT(DISTINCT {rule.conclusion.objectD}) AS ?count) WHERE {{\n"""
    else:
        raise Exception(f'only subject or object you passed {on}')

    query += f"{rule.conclusion.subject} {rule.conclusion.predicate} {rule.conclusion.objectD} ."
    return query + """\n}"""


def query_support(rule):
    # var = ?a
    # rule = ?a  place of birth  ?b and ?a  place of death  ?b

    query = f""" SELECT ( COUNT( DISTINCT * ) AS ?count ) WHERE {{\n"""
    query += f"""SELECT DISTINCT {rule.conclusion.subject} {rule.conclusion.objectD} WHERE {{"""

    for hyp in rule.hypotheses:
        query += f'\n{hyp.atom} .'

    query += f'\n{rule.conclusion.atom}'

    return query + """\n}}"""


def query_body_size(rule):  # TODO:check for  f"""SELECT DISTINCT {' '.join(rule.rule_variables)} WHERE {{"""

    query = f""" SELECT ( COUNT( DISTINCT * ) AS ?count ) WHERE {{\n"""
    query += f"""SELECT {' '.join(rule.conclusion.atom_variables)} WHERE {{"""
    for hyp in rule.hypotheses:
        query += f'\n{hyp.atom} .'
    query += """\n}}"""
    # query += f"""GROUP BY {rule.conclusion.subject} {rule.conclusion.objectD}"""
    # query+= """\n}"""
    return query


def query_pca_body_size(rule):  # TODO: join(rule_parent.rule_variables)} check replace ?a ?b
    query = f""" SELECT ( COUNT( DISTINCT * ) AS ?count ) WHERE {{\n"""
    query += f"""SELECT DISTINCT {' '.join(rule.conclusion.atom_variables)} WHERE {{"""
    for hyp in rule.hypotheses:
        query += f'\n{hyp.atom} .'

    if rule.functionalVariable == rule.conclusion.subject:
        query += f'\n{rule.conclusion.subject} {rule.conclusion.predicate} ?anything '
    elif rule.functionalVariable == rule.conclusion.objectD:
        query += f'\n?anything {rule.conclusion.predicate} {rule.conclusion.objectD}'
    return query + """\n}}"""


def query_head_size(rule):
    query = f""" SELECT ( COUNT( DISTINCT * ) AS ?count ) WHERE {{\n"""

    # query += f"""SELECT DISTINCT {rule.conclusion.subject} {rule.conclusion.objectD} WHERE {{"""
    query += f'\n{rule.conclusion.atom}'
    query += """\n}"""
    return query


def query_pca_body_size_num_pred_interval(rule_parent, base_q, pred_num, var_np, min_interval, max_interval,
                                          include_exclude):
    query = f"""SELECT ( COUNT( DISTINCT * ) AS ?count ) WHERE {{\n"""
    query += f"""SELECT DISTINCT {' '.join(rule_parent.conclusion.atom_variables)} WHERE {{"""

    query += base_q

    query += f'\n{var_np} <{pred_num}> ?x .'

    if min_interval != float(-np.inf):
        if max_interval != float(np.inf):
            if include_exclude == 'exclude':
                query += f'\nFILTER(?x >= {max_interval} || ?x < {min_interval}) .'
            else:
                query += f'\nFILTER(?x >= {min_interval} && ?x < {max_interval}) .'

        else:  # max_interval==inf
            if include_exclude == 'exclude':
                query += f'\nFILTER(?x < {min_interval}) .'
            else:
                query += f'\nFILTER(?x >= {min_interval}) .'

    else:  # min_interval == -inf
        if max_interval != float(np.inf):
            if include_exclude == 'exclude':
                query += f'\nFILTER(?x >= {max_interval}) .'
            else:
                query += f'\nFILTER(?x < {max_interval}) .'

    return query + """\n}}"""


def query_body_size_num_pred_interval(rule_parent, pred_num, var_np, min_interval, max_interval, include_exclude):
    query = f"""SELECT ( COUNT( DISTINCT * ) AS ?count ) WHERE {{\n"""
    query += f"""SELECT DISTINCT {' '.join(rule_parent.conclusion.atom_variables)} WHERE {{"""
    for hyp in rule_parent.hypotheses:
        query += f'\n{hyp.atom} .'
    query += f'\n{var_np} <{pred_num}> ?x .'
    # query += f"""GROUP BY {rule.conclusion.subject} {rule.conclusion.objectD}"""
    # query+= """\n}"""

    if min_interval != float(-np.inf):
        if max_interval != float(np.inf):
            if include_exclude == 'exclude':
                query += f'\nFILTER(?x >= {max_interval} || ?x < {min_interval}) .'
            else:
                query += f'\nFILTER(?x >= {min_interval} && ?x < {max_interval}) .'

        else:  # max_interval==inf
            if include_exclude == 'exclude':
                query += f'\nFILTER(?x < {min_interval}) .'
            else:
                query += f'\nFILTER(?x >= {min_interval}) .'

    else:  # min_interval is -inf
        if max_interval != float(np.inf):
            if include_exclude == 'exclude':
                query += f'\nFILTER(?x >= {max_interval}) .'
            else:
                query += f'\nFILTER(?x < {max_interval}) .'

    return query + """\n}}"""


def helper_select_replace(rule: Rule, head_missing):
    if head_missing:
        _select = rule.conclusion.subject
        _select_raw = rule.conclusion.subject_raw

        replace_it = rule.conclusion.objectD
        _replace_it_raw = rule.conclusion.object_raw

    else:
        _select = rule.conclusion.objectD
        _select_raw = rule.conclusion.object_raw

        replace_it = rule.conclusion.subject
        _replace_it_raw = rule.conclusion.subject_raw
    return _select, _select_raw, replace_it, _replace_it_raw


def create_query_kg_completion_numerical(parent_rule, numerical_part_dict, include_exclude, instances,
                                         head_missing=True):
    _select, _select_raw, replace_it, _replace_it_raw = helper_select_replace(parent_rule, head_missing)
    _map_dirs = {'<=': '>', '>': '<='}
    instances = [f'<{inst}>' for inst in instances]

    knows_query = f"""SELECT DISTINCT {_select} {replace_it} WHERE {{"""
    for hyp in parent_rule.hypotheses:
        knows_query += f'\n{hyp.atom} .'
    for i, k in enumerate(numerical_part_dict):
        knows_query += f'\n{k[0]} <{k[1]}> ?num{i} .'

    knows_query += f"\nFILTER( {replace_it} IN ({', '.join(instances)})) ."

    if include_exclude != 'existential':
        filter_part = ''
        for i, (k, v) in enumerate(numerical_part_dict.items()):
            s = ''
            for dir, thr in v.items():
                operator = '||' if include_exclude == 'exclude' else '&&'
                direction = _map_dirs[dir] if include_exclude == 'exclude' else dir
                s += f'?num{i} {direction} {thr} {operator} '

            s = s[:-4]

            operator = '||' if include_exclude == 'exclude' else '&&'
            filter_part += f' {s} {operator}'

        knows_query += f'\nFILTER({filter_part[:-2].strip()}) .'

    return knows_query + """\n}""", _select_raw, _replace_it_raw


def create_query_kg_completion(rule, instances, head_missing=True):
    _select, _select_raw, replace_it, _replace_it_raw = helper_select_replace(rule, head_missing)

    instances = [f'<{inst}>' for inst in instances]
    knows_query = f"""SELECT DISTINCT {_select} {replace_it} WHERE {{"""
    for hyp in rule.hypotheses:
        knows_query += f'\n{hyp.atom} .'

    knows_query += f"\nFILTER( {replace_it} IN ({', '.join(instances)})) ."

    return knows_query + """\n}""", _select_raw, _replace_it_raw


def create_query_kg_completion_intervals(rule, pred_num, var_np, instance=None, head_missing=True, global_query=False):
    global_inst, instance, _select, replace_it = helper_select_replace(rule, instance, head_missing, global_query)

    knows_query = f"""SELECT DISTINCT {_select}{global_inst} ?x WHERE {{"""
    for hyp in rule.hypotheses:
        knows_query += "\n"
        if hyp.subject == replace_it:
            knows_query += f"{str(instance)} "
        else:
            knows_query += f"{hyp.subject} "
        knows_query += f"{hyp.predicate} "
        if hyp.objectD == replace_it:
            knows_query += f"{instance} ."
        else:
            knows_query += f"{hyp.objectD} ."

    if var_np == replace_it:
        knows_query += f'\n{instance} <{pred_num}> ?x .'
    else:
        knows_query += f'\n{var_np} <{pred_num}> ?x .'

    return knows_query + """\n}""", _select.replace('?', '')


def create_query_kg_completion_intervals_(rule_parent, pred_num, var_np, min_interval, max_interval, instance,
                                          head_missing=True):
    if head_missing:
        _select = rule_parent.conclusion.subject
        replace_it = rule_parent.conclusion.objectD
    else:
        _select = rule_parent.conclusion.objectD
        replace_it = rule_parent.conclusion.subject
    knows_query = f"""SELECT DISTINCT {_select} WHERE {{"""
    for hyp in rule_parent.hypotheses:
        knows_query += "\n"
        if hyp.subject == replace_it:
            knows_query += f"{str(instance)} "
        else:
            knows_query += f"{hyp.subject} "
        knows_query += f"{hyp.predicate} "
        if hyp.objectD == replace_it:
            knows_query += f"{instance} ."
        else:
            knows_query += f"{hyp.objectD} ."

    if var_np == replace_it:
        knows_query += f'\n{instance} <{pred_num}> ?x .'
    else:
        knows_query += f'\n{instance} <{pred_num}> ?x .'
    """
    if min_interval != float(-np.inf):
        if max_interval != float(np.inf):
            knows_query += f'\nFILTER(?x >= {max_interval} || ?x < {min_interval}) .'
        else:  # max_interval==inf
            knows_query += f'\nFILTER(?x < {min_interval}) .'

    else:
        if max_interval != float(np.inf):
            knows_query += f'\nFILTER(?x >= {max_interval}) .'
    # if tail:
    #    knows_query+= f"\n{rule.conclusion.subject} {rule.conclusion.predicate} {instance} ."
    # else:
    #    knows_query+= f"\n{instance} {rule.conclusion.predicate} {rule.conclusion.objectD} ."
    """
    return knows_query + """\n}"""


def create_query_filter_kg_completion(rule, instance, tail=True):
    if tail:
        _select = rule.conclusion.subject
    else:
        _select = rule.conclusion.objectD
    query = f"""SELECT {_select} WHERE {{"""
    for hyp in rule.hypotheses:
        query += f'\n{hyp.atom} .'
    # query += f'\n{rule.conclusion} .'
    if tail:
        query += f'\nFILTER({rule.conclusion.objectD}={instance}).'
    else:
        query += f'\nFILTER({rule.conclusion.subject}={instance}).'
    query += """\n}"""
    return query
