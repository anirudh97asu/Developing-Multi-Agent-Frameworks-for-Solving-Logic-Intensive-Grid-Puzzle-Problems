import inspect
import numpy as np
import pandas as pd
import sys
import os
from itertools import combinations

sys.path.append(os.path.dirname(os.getcwd()))
from src.llm_engine import tools

def run_agent_workflow(agent_name, user_query, system_prompt, agent_job, json, Agent):
    
    instance_ = Agent(agent_name=agent_name, system_prompt=system_prompt+agent_job, json=json)

    agent_response = instance_.perform_action(user_query)

    return agent_response

def load_prompt_info(agent_name, prompt_json):
    system_prompt = prompt_json[agent_name]["system_prompt"]
    agent_job = prompt_json[agent_name]["agent_job"]
    json = prompt_json[agent_name]["json"]

    return system_prompt, agent_job, json

def get_tabular_format_from_question(question):
    res = []
    lines = question.split("\n")
    for line in lines:
        if "|" in line:
            res.append(line)
    
    return "\n".join(res)


def process_categories(categories_str):
    cat_topics = []
    categories = []
    cat_str = categories_str.split("\n")
    for item in cat_str:
        cat_topics.append(item.split(" : ")[0])
        categories.append(item.split(" : ")[1])

    return cat_topics, categories

def accumulate_tools():

    script = tools

    tool_descriptions = {}

    total_tools  = []
    functions = inspect.getmembers(script, inspect.isfunction)

    # Extract function names
    function_names = [(name,fn) for name, fn in functions if fn.__module__  == str(script.__name__)]
    total_tools.extend(function_names)

    available_tools = {k:fn for (k, fn) in total_tools}
    tool_descriptions = [f"{name}:\n{func.__doc__}\n\n" for name, func in available_tools.items()]

    return tool_descriptions

def initialize_probability_tables(num_categories, categories, cat_topics):
    """
    Initialize uniform probability matrices for all pairs of categories.
    
    Parameters:
    num_categories: int - Number of categories
    categories: list - List of category strings with comma-separated subcategories
    
    Returns:
    matrices: dict - Maps category index tuples to probability matrices
    category_mapping: dict - Maps topic strings to category index tuples
    
    Example:
    >>> matrices, mapping = initialize_probability_tables(3, ["red, blue", "small, large", "cheap, expensive"])
    """
    r = 2
    combinations_ls = list(combinations(np.arange(num_categories), r))
    matrices = {}
    for index, combination in enumerate(combinations_ls):
        c1 , c2 = categories[combination[0]].split(", "), categories[combination[1]].split(", ")
       
        matrix =  np.full((len(c1), len(c2)), 1 / max(len(c1), len(c2)))
        topic = cat_topics[combination[0]] + "__x__" + cat_topics[combination[1]]
        matrices[topic] = matrix
       
    return matrices


def populate_categories(categories):
    
    cat_topics = []
    categories_list = []


    cat_str = categories.split("\n")
    for item in cat_str:
        cat_topics.append(item.split(" : ")[0])
        categories_list.append(item.split(" : ")[1].split(", "))

    examples = {}
    r = 2
    combinations_ls = list(combinations(np.arange(len(cat_topics)), r))

    for topic_cob in combinations_ls:
        values = []
        key_1, key_2 = topic_cob
        for i in range(len(categories_list[0])):
            for j in range(len(categories_list[0])):

                values.append((categories_list[key_1][i], categories_list[key_2][j]))

        examples[cat_topics[topic_cob[0]] + "__x__" + cat_topics[topic_cob[1]]] = values

    return examples


# def get_pair_index_mapping(pair_ind):
    
#     mapping = {0: (0, 0), 1: (0, 1), 2: (0, 2), }