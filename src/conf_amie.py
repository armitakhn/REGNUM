
# run_command_amie = f'java -jar {self.path_amie3} {self.path_save_rules + "train_not_num.txt"} {"-htr is_city"} -const {"-minhc " + str(self.min_hc)} {"-minc " + str(self.min_conf)} {"-maxad " + str(self.num_atoms)}'


MAX_NUM_ATOMS = 3
MIN_CONF = 0.05
MIN_HC = 0.05

BINNING_TECHNIQUE = "opt"

# conclude on a head relation
HTR = "is_city"
CONST = False


