
from pathlib import Path
import json
import re
import os
import pickle
import codecs
from typing import Dict, List, Any


def is_exist(path):
    path_obj = Path(path)
    return path_obj.exists()


def get_logger(filename, clear=False):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if clear:
        co = codecs.open(filename, 'w', 'utf-8')
        co.close()

    def logger(string):
        with codecs.open(filename, 'a', 'utf-8') as output:
            output.write(string + "\n")

    return logger


def get_filenames_under_directory(path: str, extention: str) -> List[str]:
    filenames = []
    for file in os.listdir(path):
        index = re.match(r'.*\.{}'.format(extention), file)
        if index and not file.startswith('.'):
            filenames.append(file)
    return filenames




