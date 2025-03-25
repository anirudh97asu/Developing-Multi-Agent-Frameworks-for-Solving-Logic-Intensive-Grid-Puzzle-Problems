import os
import logging
import json

import traceback
from tqdm import tqdm
from src.llm_engine.gemini_agent import Agent
from src.utils import *
from src.llm_engine.tools import *
from src.llm_engine.scheduler import scheduler
from pprint import pprint

__name__ = "__run_agentic_framework__"
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

from src.utils import accumulate_tools

def run_multi_agent_system(df, index, prompt_json, Agent=None):

    agent_response = None

    if Agent is not None:

        question = df["question"].iloc[index]
        clues = df["Clues"].iloc[index]
        categories_str = df["categories"].iloc[index]
        user_query = clues + "\n" + categories_str

        cat_topics,categories = process_categories(categories_str=categories_str)

        for index, agent in enumerate(prompt_json):
            

            if index == 0:
                system_prompt,agent_job, json_ = load_prompt_info(agent_name=agent, prompt_json=prompt_json)

                # agent 1 response 
                agent_response = run_agent_workflow(agent,
                                                        user_query=user_query,
                                                        system_prompt=system_prompt,
                                                        agent_job=agent_job,
                                                        json=json_,
                                                        Agent=Agent)
                        
                
            if index == 1:

                system_prompt,agent_job, json_ = load_prompt_info(agent_name=agent, prompt_json=prompt_json)
                
                probability_matrices = initialize_probability_tables(num_categories=len(categories),
                                                                     categories=categories,
                                                                     cat_topics=cat_topics
                                                                     )
                user_query = agent_response
                agent_job += f"Logical Step-By-Step-Deductions: {agent_response}\n"
                agent_job += f"Initial Probability Matrices: {probability_matrices}\n"

                # agent 2 response
                agent_response = run_agent_workflow(agent,
                                                    user_query=agent_job,
                                                    system_prompt=system_prompt,
                                                    agent_job="",
                                                    json=json_,
                                                    Agent=Agent
                                                    )
                
            elif index == 2:

                system_prompt,agent_job, json_ = load_prompt_info(agent_name=agent, prompt_json=prompt_json)
                tool_descriptions = accumulate_tools()
                user_query = f"The categories are: {categories}\n And here are the probability updates: {agent_response}"
                system_prompt += "".join(tool_descriptions) + "\n"

                # agent 3 response
                agent_response =run_agent_workflow(agent,
                                                    user_query=user_query,
                                                    system_prompt=system_prompt,
                                                    agent_job=agent_job,
                                                    json=json_,
                                                    Agent=Agent
                                                    )
            
            elif index == 3:
                tabular_format = get_tabular_format_from_question(question)

                calls = json.loads(agent_response)
                for index, call in enumerate(calls):
                    fn_ = call["tool_name"]
                    if fn_ == "update_matrix":
            
                            updated_matrices = update_matrix(probability_matrices, matrix_name=call["matrix_name"],
                                                row=int(call["i"]), col=int(call["j"]),prob_value=float(call["value"]))
                            
                            pprint(updated_matrices)
                            print("End of Update {}".format(index))
                            

                system_prompt,agent_job, json_ = load_prompt_info(agent_name=agent, prompt_json=prompt_json)
                system_prompt += tabular_format + "\n"
                agent_job += f"The matrices are: {updated_matrices}\n"
                agent_job += f"The categories are: {categories}\n"

                
                # agent 4 response
                agent_response =run_agent_workflow(agent,
                                                    user_query=agent_job,
                                                    system_prompt=system_prompt,
                                                    agent_job="",
                                                    json=json_,
                                                    Agent=Agent
                                                    )
            
                
    return agent_response
        


if __name__ == "__run_agentic_framework__":

    logger.info("Starting agentic Framework")
    
    logger.info("""
        ╔═══════╗
        ║ ◣_◢ ║
        ║ {○_○} ║
    ╔═══╩═══════╩═══╗
    ║  [│_█_█_█_│]  ║
    ╚═══╦═══════╦═══╝
        ║  ═══  ║
        ║   │   ║
        ╚═══╦═══╝
            ║
         ═══╩═══        
    """)

    try:

        pass

    
    except Exception as e:
        logger.info(f"There is some issue in running your process: {str(traceback.format_exc(e))}")

        
