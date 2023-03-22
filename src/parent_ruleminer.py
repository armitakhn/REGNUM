from abc import abstractmethod, ABC

from rule import Rule, Atom
from subprocess import check_output
import pandas as pd
import os
from quality_measures import RuleMeasure
from tqdm import tqdm
from utils import create_tmp_folder

class RunParseBase:
    def __init__(self, data, path_rule_miner, path_save_rules, num_atoms, min_conf, force=False,
                 const=False, default=True):
        self.data: pd.DataFrame = data
        self.path_rule_miner: str = path_rule_miner
        self.path_save_rules: str = path_save_rules

        self.num_atoms: int = num_atoms
        self.min_conf: float = min_conf

        self.force_create_rule = force
        self.const: bool = const
        self.default: bool = default
        self._data_input_save_to_file()

    def _data_input_save_to_file(self):
        # df_num_mine_rules_on = self.dataloader.df_not_num.copy()
        df_num_mine_rules_on = self.data.copy()
        # df_num_mine_rules_on = df_num_mine_rules_on.apply(lambda x: '<http:'+x+'>')
        create_tmp_folder(self.path_save_rules)
        df_num_mine_rules_on.to_csv(self.path_save_rules + "/train.txt", sep="\t", header=False, index=False)

    @abstractmethod
    def run(self):
        ...

    @abstractmethod
    def parse(self, *args):
        ...


class RunParseAMIE(RunParseBase):
    def __init__(self, data, path_rule_miner, path_save_rules, num_atoms=None,
                 min_conf=None, min_pca_conf=None, min_hc=None, force=False, const=False, default=True):
        super().__init__(data, path_rule_miner, path_save_rules, num_atoms, min_conf, force, const, default)

        self.min_pca_conf: float = min_pca_conf

        self.min_hc: float = min_hc
        self.mineRules = True if not os.path.exists(self.path_save_rules + "/amierules_main.txt") else False

        if self.mineRules or self.force_create_rule:
            self.run()
            self.rules_mined_f = self.res_rules_raw.decode("utf-8").split("\n")
        else:
            print("loading from file...")
            self.rules_mined_f = open(self.path_save_rules + "/amierules_main.txt", "r")

    def run(self):

        if self.const:
            print('mining rules with const')
            # run_command_amie = f'java -jar {self.path_rule_miner} {self.path_save_rules + "train_not_num.txt"} {"-htr" + HTR} -const {"-minhc " + str(self.min_hc)} {"-minc " + str(self.min_conf)} {"-maxad " + str(self.num_atoms)}'
            run_command_amie = f'java -jar {self.path_rule_miner} {self.path_save_rules + "train.txt"} -const {"-minhc " + str(self.min_hc)} {"-minc " + str(self.min_conf)} {"-minpca " + str(self.min_pca_conf)} {"-maxad " + str(self.num_atoms)}'
        else:
            run_command_amie = f'java -jar {self.path_rule_miner} {self.path_save_rules + "train.txt"} {"-minhc " + str(self.min_hc)} {"-minc " + str(self.min_conf)} {"-minpca " + str(self.min_pca_conf)} {"-maxad " + str(self.num_atoms)}'
        if self.default:
            if self.const:
                print('default true, const true')
                run_command_amie = f'java -jar {self.path_rule_miner} {self.path_save_rules + "train.txt"} -const'

            else:
                run_command_amie = f'java -jar {self.path_rule_miner} {self.path_save_rules + "train.txt"}'

        print(f'default is {self.default} {run_command_amie}')

        self.res_rules_raw = check_output(run_command_amie, shell=True)

        # print(len(self.res_rules_raw))
        file = open(self.path_save_rules + "/amierules_main.txt", "w")
        for line in self.res_rules_raw.decode("utf-8").split("\n"):
            file.write(str(line) + "\n")
        file.close()

    def parse_amie(self):
        print('use parse() instead')
        return self.parse()

    def parse(self):
        rules = []
        for line in self.rules_mined_f:
            if (line != "") and (line[0] == "?"):
                rules.append(Rule(line))

        print("number of mined rules:", len(rules))
        return rules


class RunParseAnyBurl(RunParseBase):
    def __init__(self, data, path_rule_miner, path_save_rules, num_atoms, min_conf,
                 batch_time=5000, force=False, const=False, default=True):
        super().__init__(data, path_rule_miner, path_save_rules, num_atoms, min_conf, force,
                         const, default)

        self.batch_time = str(batch_time)

        self.path_saved_rules = f'{self.path_save_rules}anyburl_rule-{self.batch_time}'
        self.mineRules = True if not os.path.exists(self.path_saved_rules) else False
        print(self.path_saved_rules, self.mineRules)
        if self.mineRules or self.force_create_rule:
            self.run()

    def run(self):
        base = self.path_rule_miner.rsplit('/', 1)[0]
        path_prop = base + '/config-learn.properties'
        self._write_properties(path_prop)
        r2 = f'java -Xmx12G -cp {self.path_rule_miner} de.unima.ki.anyburl.LearnReinforced {path_prop}'
        _ = check_output(r2, shell=True)

    def _write_properties(self, path_prop):

        file = open(path_prop, "w")
        a = f'PATH_TRAINING = {self.path_save_rules}train.txt'
        b = f'PATH_OUTPUT = {self.path_save_rules}anyburl_rule'
        c = f'SNAPSHOTS_AT = {self.batch_time}'
        d = 'THRESHOLD_CORRECT_PREDICTIONS = 10'
        e = f'BATCH_TIME = {self.batch_time}'
        f = 'WORKER_THREADS = 4'
        g = 'CONSTANTS_OFF = true'
        h = f'THRESHOLD_CONFIDENCE = {self.min_conf}'
        ii = 'MAX_LENGTH_ACYCLIC = 0'
        for i in [a, b, c, d, e, f, g, h, ii]:
            file.write(i + "\n")
        file.close()

    def parse(self, graph):

        rules_mined_f = open(self.path_saved_rules, "r")
        rules = []
        i = 1

        for line in tqdm(rules_mined_f):

            head, body = self._prepare_rule(line)
            if self.rule_head_constant(head):
                continue
            if self.rule_is_not_closed(head, body):
                continue

            if self.rule_more_than_max_atoms(body):
                continue

            amie_like_line = self.build_amie_like_line(body, head)
            rule = Rule(amie_like_line)

            rule = self.set_rule_attributes(rule, graph)
            rules.append(rule)
            i += 1

        print("number of mined rules:", len(rules))
        return rules

    def rule_more_than_max_atoms(self, body_list):
        if len(body_list) > self.num_atoms - 1:
            return True
        return False

    def set_rule_attributes(self, rule, graph):
        measures = RuleMeasure(rule, graph)

        rule.set_functionalVariable(measures.func_var)
        rule.set_headCoverage(measures.head_coverage)
        rule.set_stdConfidence(measures.standard_confidence)
        rule.set_pcaConfidence(measures.pca_confidence)
        rule.set_support(measures.support)
        rule.set_bodySize(measures.body_size)
        rule.set_pcaBodySize(measures.pca_body_size)

        return rule

    def build_amie_like_line(self, body_list, head):
        line = ''
        for body in body_list:
            line += f'?{body[0]}  {body[1]}  ?{body[2]}  '
        line += ' => '
        line += f'?{head[0]}  {head[1]}  ?{head[2]}'
        return line

    def rule_is_not_closed(self, head, body_list):
        occ = {head[0]: 1, head[2]: 1}
        for body in body_list:
            if body[0] in occ:
                occ[body[0]] += 1
            else:
                occ[body[0]] = 1

            if body[2] in occ:
                occ[body[2]] += 1
            else:
                occ[body[2]] = 1

        occ_ = list(set(occ.values()))
        if occ_[0] == 2 and len(occ_) == 1:
            return False
        return True

    def rule_head_constant(self, head):  # NOT SURE THIS IS THE BEST WAY, WHAT IF ITS XX OR YY
        if len(head[2]) > 1 or len(head[0]) > 1:
            return True
        return False

    def _prepare_rule(self, line):

        _rule = line.split('\t')[3]
        _rule = _rule.strip()
        head, body = _rule.split('<=')

        body_list = body.split('),')
        body_list = [body.strip() for body in body_list]
        body_list = [body + ')' if body[-1] != ')' else body for body in body_list]

        body_triples = []
        for body in body_list:
            body_triples.append(self._one_set_parse(body))

        return self._one_set_parse(head), body_triples

    def _one_set_parse(self, element):
        prop, head_rest = element.split('(')
        head_rest = head_rest.strip()[:-1]
        subj, obj = head_rest.split(',')
        return subj, prop, obj
