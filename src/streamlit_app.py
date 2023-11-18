import os
import pickle
import streamlit as st
from datetime import datetime

class StreamlitRunner():
    def __init__(self):
        self._page_decs()
    

    def _page_decs(self):

        st.set_page_config(layout="wide", page_icon=":mag_right:", page_title="REGNUM")
        st.title(":mag_right: REGNUM")
        st.markdown(f"![image](https://badgen.net/badge/version/{os.getenv('VERSION', 'test')}/green)")
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    def write(self, *comment):
        st.write(*comment)
    def warning(self, comment):
        st.warning(comment)
    def success(self, comment):
        st.success(comment)
    def title(self, comment):
        st.title(comment)
    
    def log(self, success, comment):
        if success in [True, 'OK']:
            self.success(comment)
        else:
            self.warning(comment)


    def upload_file(self):
        col1, col2 = st.columns(2)
        with col1:
            st.title('Upload the triple .tsv file')
            f = st.file_uploader("Choose a tsv file", accept_multiple_files=False, key=1)
            if f is not None:
                f_bool = True
            else:
                f_bool = False

        with col2:
            st.title('Upload the numerical predicates')
            f_num = st.file_uploader("Choose a tsv file", accept_multiple_files=False, key=2)
            if f_num is not None:
                f_num_bool = True
            else:
                f_num_bool = False
        
        return f_bool, f, f_num_bool, f_num

    def store_state(self, name, to_store):
        st.session_state[name] = to_store

    def retreve_state(self, name):
        if name in st.session_state:
            return st.session_state[name]
        else:
            return None

    @property
    def data_to_upload(self):
        if 'data_to_upload' in st.session_state and st.session_state['data_to_upload']!='':
            return st.session_state['data_to_upload']
        else:
            return None

    def button(self, note, state_name, disabled=False): 

        def make_button_visible():
            st.session_state[state_name] = True
        
        _clicked = st.button(note, on_click=make_button_visible, key=f'mapper_{note}', disabled=disabled)

        clicked = state_name in st.session_state and st.session_state[state_name]
        return clicked
    
    def show_sample(self, sheet, top_n=6):
        self.write(f'Sample first {top_n} lines', sheet.head(top_n))
    
    def expander_pick_value(self, label, keys, options):
        st.title(label)
        map_col_dict = {}
        my_expander = st.expander(label=label, expanded=True)

        with my_expander:
            for col in keys:
                sorted_options = options
                option = st.selectbox(
                f'Select for {col}',sorted_options)

                map_col_dict[col] = option
            
            return map_col_dict
            #if clicked:
            #    return True, map_col_dict
            #return False, map_col_dict

    def download_list_json(self, note, filename, data:list):
        # Add input widgets to the columns
        clicked = st.download_button(
        note+' pickle format',
        data=pickle.dumps(data),
        file_name=filename+'.pkl',
    )

        return clicked

    def expander_set_value(self, label, keys):
        st.title(label)
        map_col_dict_option = {}
        map_col_dict_select = {}

        my_expander = st.expander(label=label, expanded=True)

        with my_expander:
            for col in keys:
                col1, col2 = st.columns(2)
        # Add input widgets to the columns
                with col1:
                    option = st.text_input(
                    f'Set dataimpact retailer id for `"{col}"` (int)', 'remove me', key=f'key_{col}')
                with col2:
                    off_on = st.selectbox(f'Select for {col}',['online', 'offline'])


                map_col_dict_option[col] = option if option=='remove me' else int(option)
                map_col_dict_select[col] = off_on

            self.log(True, f'mapping: {map_col_dict_option, map_col_dict_select}')
            return map_col_dict_option, map_col_dict_select
            #if clicked:
            #    return True, map_col_dict
            #return False, map_col_dict
    def celebrate(self):
        st.balloons()

from data_loader import GeneralDataLoader
from graph_data import RDFLibGraph, StarDogGraph
from parent_ruleminer import RunParseAMIE
from runner import run
import random
import pandas as pd
import json
from rule_writer import CustomJsonEncoder
def main():
    f_name= 'DB15K_num'
    base = '/Documents/project/REGNUM/data'
    p = f'{base}/datasets/{f_name}'
    PATH_RM = f"{base}/rule_miners/amie_jar/amie3.jar"
    PATH_result = f"{base}/results/{f_name}/"
    path_save_enriched_rules = f"{base}/results/{f_name}/enriched_rules.txt"

    sr = StreamlitRunner()
    if sr.retreve_state('log') == None:
        file_bool, f_file, numerical_bool, f_num = sr.upload_file()
        date_time = datetime.now().strftime("%m/%d/%Y_%H:%M:%S")
        if not file_bool or not numerical_bool:
            return
        dl = GeneralDataLoader(path_t=f_file, path_numerical_preds=f_num)

        sr.success('file Successfully Uploaded!')
        sr.show_sample(dl.df)
        rule_base = sr.expander_pick_value(label='Choose Base rule miner', keys=['rule_miner'], options=['','Amie', 'AnyBurl'])
        query_engine = sr.expander_pick_value(label='Choose Base query engine', keys=['query_engine'], options=['','RDFLib', 'StarDog'])

        if rule_base['rule_miner'] != 'Amie':
            return
        if query_engine['query_engine'] != 'RDFLib':
            return

        clicked_prepare_upload = sr.button(note='Click to Mine Rules', state_name='prepare_upload')
        if not clicked_prepare_upload:
            return
        
        sr.title('Constructing Graph and Base rules')

        gr = RDFLibGraph(dl, database_name='sample', force=True, p_save_g=PATH_result+'graph.ttl')
        amie = RunParseAMIE(data=dl.df, path_rule_miner=PATH_RM,
                    path_save_rules=PATH_result)
        rules = amie.parse()
        sr.success(f'{len(rules)} rules have been generated using {rule_base["rule_miner"]} as base rule')
        sr.show_sample(pd.DataFrame({'few sample rules': rules[37:40]}))

        dict_all_new_rules_f_score = run(rules[37:40], gr, list(dl.numerical_preds))

        json_string = json.dumps(dict_all_new_rules_f_score, indent=4, cls=CustomJsonEncoder)


        st.json(json_string, expanded=True)

        st.download_button(
        label="Download REGNUM Rules",
        file_name="REGNUM_rules.json",
        mime="application/json",
        data=json_string,
        )
            

if __name__ == "__main__":
    main()