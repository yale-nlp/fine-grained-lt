import numpy as np
import re

from questeval.questeval_metric import QuestEval
from typing import List, Dict

def calculate_questeval(sources: List[str], 
                        predictions: List[str], 
                        labels: List[List[str]], 
                        questeval, 
                        both=True):
    """_summary_

    Args:
        source (list[str]): List of input sources
        prediction (list[str]): List of output sources
        labels (list[list[str]]): List of list of reference strings

    Returns:
        dict: Output of computed metrics
    """
    result = {}
    if both:
        score = questeval.corpus_questeval(
            hypothesis=predictions, sources=sources, list_references=labels
        )
        result["questeval_ref"] = score["corpus_score"]
        result["questeval_ref_std"] = np.std(score["ex_level_scores"])

    score = questeval.corpus_questeval(
        hypothesis=predictions,
        sources=sources,
    )
    result["questeval_no_ref"] = score["corpus_score"]
    result["questeval_no_ref_raw"] = score["ex_level_scores"]
    result["questeval_no_ref_std"] = np.std(score["ex_level_scores"])

    return result

def clean_string(s: str):
    s = s.replace("-lrb-"," ").replace("-rrb-", " ")
    s = s.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
    return re.sub(" +", " ", s)

def compute_metrics(
    sources: List[str],
    predictions: List[str],
    labels: List[List[str]],
    metrics: List[str],
) -> Dict:
    """Test docstring.

    Args:
        sources (list[str]): List of input sources
        predictions (list[str]): List of output sources
        labels (list[list[str]]): List of list of reference strings
    Returns:
        dict: Output of computed metrics

    """
    assert type(sources) == list and type(sources[0]) == str, print(
        "Sources should be a list of strings"
    )
    assert type(predictions) == list and type(predictions[0]) == str, print(
        "Predictions should be a list of strings"
    )
    assert type(labels) == list and type(labels[0]) == list, print(
        "Labels should be a list of LISTS, each containing the labels"
    )

    # Clean inputs
    sources = [clean_string(s) for s in sources]
    predictions = [clean_string(s) for s in predictions]
    labels = [[clean_string(s) for s in lst] for lst in labels]

    result = {}

    if "questeval" in metrics:
        questeval = QuestEval(no_cuda=False, use_cache=True)
        questeval_dict = calculate_questeval(sources, predictions, labels, questeval)
        result.update(questeval_dict)

    return {k: round(v, 4) if type(v) in [float, int] else v for k, v in result.items()}
