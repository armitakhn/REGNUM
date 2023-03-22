#!flask/bin/python

import logging
from flask import Flask, request, json
import warnings
import main_ws

warnings.filterwarnings('ignore')
app = Flask(__name__)
app.logger.setLevel(logging.INFO)


class BadRequest(Exception):
    pass


def parse_content(req, fields):
    try:
        content = req.get_json()
        print(content)
        params = [content[field] for field in fields]

        return params

    except Exception:
        raise BadRequest('Bad request format, please fix the request format and try again.')


@app.route("/miner", methods=['POST'])
def rule_miner():
    path_t, path_numerical_preds, num_atoms, min_conf, min_hc = parse_content(request, ['KB_path', 'num_predicate_path',
                                                                                        'num_atoms', 'min_conf',
                                                                                        'min_hc'])

    res_json = main.main_enricher_ws(path_t, path_numerical_preds, num_atoms, min_conf, min_hc)
    #res = {'aa': 'qq'}
    return res_json
    #return json.dumps(res)


@app.before_first_request
def create_folders_paths():
    main.prepare_path_env()
    app.logger.info("data folders created")


@app.route("/health_check", methods=['GET'])
def health_check():
    return 'EVERYTHING IS OKAY'


if __name__ == "__main__":
    app.run(debug=False, port=1030, host='0.0.0.0')
