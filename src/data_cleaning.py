import pandas as pd
import csv


def clean_wiki_num_data(path):
    train_path = path + "train.txt"
    numericals_path = path + "numeric_literals.txt"

    pd_train = pd.read_csv(train_path, sep="\t", index_col=False, names=["subject", "predicate", "object"])
    pd_numericals = pd.read_csv(numericals_path, sep="\t", index_col=False, names=["subject", "predicate", "object"])

    df_all = pd.concat([pd_train, pd_numericals], ignore_index=True)
    df_all.to_csv(path + "train_full.txt", sep="\t", header=False, index=False)

    f = open(path + "numerical_preds.txt", "w")
    for pred in pd_numericals.predicate.unique():
        f.write(pred)
        f.write("\n")
    f.close()


def merge_to_labels(*args, path_save):
    pd_labels = []
    for path in args:
        pd_labels.append(pd.read_csv(path, sep="\t", index_col=False, header=None))

    df_all_to_label = pd.concat(pd_labels)
    df_all_to_label.to_csv(path_save, sep="\t", header=False, index=False)


if __name__ == "__main__":
    # clean_wiki_num_data("../data/LiterallyWikidata/LitWD19K/")
    merge_to_labels("../data/LiterallyWikidata/Attributes/attribute_labels_en.txt",
                    "../data/LiterallyWikidata/Relations/relation_labels_en.txt",
                    path_save="../data/datasets/LiterallyWikidata/Predicates/predicates_labels_en.txt")
