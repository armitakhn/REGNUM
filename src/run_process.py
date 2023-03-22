import argparse, os

from data_loader import GeneralDataLoader
from graph_data import StarDogGraph
from parent_ruleminer import RunParseAMIE
from tqdm import tqdm
from runner import run
from utils import create_tmp_folder
from rule_writer import CustomJsonEncoder
import json

def parse_input():
    parser = argparse.ArgumentParser(
        description="Enrich rules"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        help="path to train tsv file",
    )
    parser.add_argument(
        "--numerical_path",
        type=str,
        help="psth to list numerical tsv path"
    )
    parser.add_argument(
        "--path_RM",
        type=str,
        help="path to jar file for rule miner (amie or anyburl)",
        default='data/rule_miners/amie_jar/amie3.jar'
    )
    parser.add_argument(
        "--path_result",
        type=str,
        help="place to store all results",
        default='data/results'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_input()
    tr_path = args.train_path
    db_name = tr_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    path_result = f'{args.path_result}/{db_name}/'
    create_tmp_folder(path_result)

    dl = GeneralDataLoader(path_t=tr_path, path_numerical_preds=args.numerical_path)
    gr = StarDogGraph(dl, database_name=db_name, force=False, p_save_g=f'{path_result}graph.ttl')
    miner = RunParseAMIE(data=dl.df, path_rule_miner=args.path_RM,
                         path_save_rules=path_result)
    rules = miner.parse()
    enriched_rules = run(rules, gr, list(dl.numerical_preds))
    with open(f'{path_result}enriched_rules.json', 'w', encoding='utf-8') as f:
        json.dump(enriched_rules, f, indent=4, cls=CustomJsonEncoder)


if __name__ == "__main__":
    main()

"""
python src/run_process.py --train_path path/to/train.tsv --numerical_path path/to/num.tsv --path_RM path/to/rm.jar --path_result path/to/result

"""
