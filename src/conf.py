import os


_CURRENT_PATH = os.getcwd()
ENDPOINTSPARQL = "https://query.wikidata.org/sparql"

### FOR WIKIDATA DATASET ###
PATH_DATA = f'{_CURRENT_PATH}/../data'
PATH_DATA_WIKI = f'{PATH_DATA}/LiterallyWikidata'
PATH_DATA_FB = f'{PATH_DATA}/FB15K_num'
PATH_PRED_LABEL_WIKI = f"{PATH_DATA_WIKI}/Predicates/predicates_labels_en.txt"
PATH_ENT_LABEL_WIKI = f"{PATH_DATA_WIKI}/Entities/entity_labels_en.txt"
PATH_ENT_TYPE_WIKI = f"{PATH_DATA_WIKI}/Entities/entity_types.txt"

PATH_RESULT = f'{_CURRENT_PATH}/../data/results'
PATH_SAVE_GRAPH = f'{PATH_RESULT}/graph.ttl'
PATH_SAVE_NEW_RULES = f'{PATH_RESULT}/enriched_rules.txt'
PATH_SAVE_AMIE_RULE = f'{PATH_RESULT}/amie_rules/'

# ########## ------------- #############
AMIE_PATH = f'{_CURRENT_PATH}/../data/rule_miners/amie_jar'
PATH_AMIE_3 = f'{AMIE_PATH}/amie3.jar'
