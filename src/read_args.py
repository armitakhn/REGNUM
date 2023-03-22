import argparse
import conf
import conf_amie


def read_args():
    parser = argparse.ArgumentParser(
        description="with numerical predicates on the shoulders of AMIE for classification")

    # Mandatory Paths#
    parser.add_argument("--path", '--p', help="set the path to train")
    parser.add_argument("--numerical_preds_path", '--npp', help="set the path to numerical predicates")

    # OPTIONAL Paths#
    parser.add_argument("--path_save_rules", '--psr', help="path_to_save_amie_rules", default=conf.PATH_SAVE_AMIE_RULE)
    parser.add_argument("--path_save_enriched_rules", '--pser', help="path_to_save_amie_rules", default=conf.PATH_SAVE_NEW_RULES)

    parser.add_argument("--pred_to_label", '--ptl', help="set the path to predicates to labels",
                        default=conf.PATH_PRED_LABEL_WIKI)
    parser.add_argument("--ent_to_label", '--pel', help="set the path to entities to predicates",
                        default=conf.PATH_ENT_LABEL_WIKI)
    parser.add_argument("--ent_to_type", '--ett', help="set the path to entities to types", default=conf.PATH_ENT_TYPE_WIKI)


    ## OPTIONAL - AMIE SETTINGS ##
    parser.add_argument("--path_amie", '--pa', help="path_to_amie_jar", default=conf.PATH_AMIE_3)

    parser.add_argument("--num_atoms_amie", '--naa', help="num_units_loss", type=int, default=conf_amie.MAX_NUM_ATOMS)
    parser.add_argument("--min_conf", '--mc', help="minimum_confidence_amie", type=int, default=conf_amie.MIN_CONF)
    parser.add_argument("--min_head_coverage", '--mhc', help="minimum_head_coverage_amie", type=int,
                        default=conf_amie.MIN_HC)

    """
    parser.add_argument("--path", '--p', help="set the path to train", required=True, choices=['train', 'eval', 'train_eval'])
    """
    return parser.parse_args()
