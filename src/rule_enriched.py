from rule import Rule


class ConstructRules:
    def __init__(self, rule, numerical_part, support, pca_body_size, pca_confidence, include_exclude,
                 head_coverage=None, f_score=None, level=None):

        self.parent_rule: Rule = rule
        self.numerical_part = self._construct_existential_numerical_part(numerical_part) if include_exclude=='existential' else numerical_part
        self.support = support
        self.pca_body_size = pca_body_size
        self.pca_confidence = pca_confidence
        self.include_exclude = include_exclude
        self.head_coverage = self.support / self.parent_rule.size_head_r if not head_coverage else head_coverage
        self.f_score = 2 * (self.pca_confidence * self.head_coverage) / (
                    self.pca_confidence + self.head_coverage) if not f_score else f_score
        self.functionalVariable = self.parent_rule.functionalVariable
        self.level = level

    def _construct_existential_numerical_part(self, d):
        var_pred_dict = {}
        for var, preds in d.items():
            for pred in preds:
                var_pred_dict[(var, pred)] = 'existential'
        return var_pred_dict

    def build_dict(self):
        d = dict()
        d['parent_rule'] = self.parent_rule
        d['include_exclude'] = self.include_exclude
        d['numerical_part'] = list(self.numerical_part)
        d['support'] = self.support
        d['pca_body_size'] = self.pca_body_size
        d['pca_confidence'] = self.pca_confidence
        d['head_coverage'] = self.head_coverage
        d['f_score'] = self.f_score
        d['level'] = self.level

        return d

    def create_raw_rule(self):
        if self.include_exclude == 'existential':
            str_num_part = self._str_num_part_existential()
        else:
            str_num_part = self._str_num_part()

        raw = f'{str_num_part} {self.parent_rule}'
        raw += f'\t{self.head_coverage}\t{self.pca_confidence}\t{self.support}\t{self.pca_body_size}\t{self.f_score}\t{self.functionalVariable}'
        return raw.strip()

    def _str_num_part_existential(self):
        s = ''
        i = 0
        for k, v_l in self.numerical_part.items():
            for v in v_l:
                s += f' {k} {v} ?num_{i}'
                i += 1
        return s

    def _str_num_part(self):
        inc_ex = '¬' if self.include_exclude == 'exclude' else ''
        s = ''
        for i, (k, v) in enumerate(self.numerical_part.items()):
            s += f' {k[0]} {k[1]} ?num_{i} '
            if len(v) == 2:
                sm = v['>']
                bg = v['<=']
                s += f'{sm} {inc_ex}<= ?num_{i} {inc_ex}< {bg}'
            else:
                rel = list(v.keys())[0]
                val = v[rel]
                s += f'?num_{i} {inc_ex}{rel} {val}'
        return s
