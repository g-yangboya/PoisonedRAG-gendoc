import sys, os
from .contriever_src.contriever import Contriever
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json
import numpy as np
from collections import defaultdict
import random
import torch
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer

model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def load_models(model_code, model_path=None):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    model_name_or_path = model_path or model_code_to_qmodel_name[model_code]
    local_only = model_path is not None
    if local_only and not os.path.isdir(model_name_or_path):
        raise FileNotFoundError(
            f"Local retriever model directory does not exist: {model_name_or_path}"
        )
    if 'contriever' in model_code:
        model = Contriever.from_pretrained(
            model_name_or_path, local_files_only=local_only)
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, local_files_only=local_only)
        get_emb = contriever_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_name_or_path)
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError
    
    return model, c_model, tokenizer, get_emb

def load_beir_datasets(dataset_name, split, datasets_dir=None):
    assert dataset_name in ['nq', 'msmarco', 'hotpotqa']
    if dataset_name == 'msmarco': split = 'train'
    if datasets_dir:
        supplied_path = os.path.abspath(os.path.expanduser(datasets_dir))
        # Also accept the former dataset-root/qrels forms for compatibility.
        data_path = supplied_path
        if os.path.basename(os.path.normpath(data_path)) == 'qrels':
            data_path = os.path.dirname(data_path)
        elif os.path.basename(os.path.normpath(data_path)) != dataset_name:
            data_path = os.path.join(data_path, dataset_name)
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"BEIR dataset directory does not exist: {data_path}")
        required = [
            os.path.join(data_path, 'corpus.jsonl'),
            os.path.join(data_path, 'queries.jsonl'),
            os.path.join(data_path, 'qrels'),
        ]
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Incomplete BEIR dataset at {data_path}; missing: {', '.join(missing)}"
            )
    else:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
        out_dir = os.path.join(os.getcwd(), "datasets")
        data_path = os.path.join(out_dir, dataset_name)
        if not os.path.exists(data_path):
            data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        split = 'train'
    corpus, queries, qrels = data.load(split=split)    

    return corpus, queries, qrels

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_results(results, dir=None, file_name="run", base_dir="results/attack_eval"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    output_dir = os.path.join(base_dir, dir) if dir else base_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_results(file_name):
    with open(os.path.join('results', file_name), encoding='utf-8') as file:
        results = json.load(file)
    return results

def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_json(file_path):
    with open(file_path, encoding='utf-8') as file:
        results = json.load(file)
    return results

def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    if len(s)>1 and s[-1] == ".":
        s=s[:-1]
    return s.lower()

def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.
    
    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.
    
    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    
    return f1_scores
