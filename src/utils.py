import json
import re
from typing import Dict

import yaml


def parse_config(config_file: str, verbose: bool = True):
    with open(config_file) as file_:
        config = yaml.load(file_, Loader=yaml.Loader)

    for key in config:
        if not isinstance(config[key], str):
            continue
        config[key] = _replace_token(config[key], config)

    if verbose:
        print("### Config ###")
        print(json.dumps(config, indent=4))

    return config


def _replace_token(string: str, config: Dict):
    # e.g. Replace "${RESULT_DIR}/model.pt" by "result_dir/model.pt".
    result = re.sub(
        r"\${[a-zA-Z0-9_]+}",
        lambda matched: _replace_token(config[matched.group()[2:-1]], config),
        string,
    )
    return result
