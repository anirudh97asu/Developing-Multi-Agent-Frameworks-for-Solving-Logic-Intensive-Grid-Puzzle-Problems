import os
import warnings
import json
import inspect

import logging
from pprint import pprint
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Dict, Union, Optional


from src.llm_engine.gemini_agent import Agent  


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("scheduler")

def accumulate_tools():
    # total_tools = []
    # for script in [organize_files, image_compression, pdf_compression, run_to_do_tasks]:
    #     functions = inspect.getmembers(script, inspect.isfunction)
    #     # Extract function names
    #     function_names = [(name,fn) for name, fn in functions if fn.__module__  == str(script.__name__)]
    #     total_tools.extend(function_names)
    
    # available_tools = {k:fn for (k, fn) in total_tools}
    # tool_descriptions = [f"{name}:\n{func.__doc__}\n\n" for name, func in available_tools.items()]
    pass


# let the agent the agent decide on how to proceed with the task
def get_list_of_steps_to_perform_user_query(user_query):

    system_prompt = f"""
        You are an helpful AI-assistant and always respond with a JSON object that has two required keys.
        task: str = The user-query given by the user.
        sub_tasks: List

        Your objective is to break down the task given by the user into multiple sequential sub-tasks. Think about solving the tasks
        from all perspectives if it's open-ended.

        Don't start your answers with "Here is the JSON response", just give the JSON.
        """
    problem_solver_agent = Agent(system_prompt=system_prompt)
    
    llm_output = problem_solver_agent.perform_action(user_query)

    return llm_output

def get_list_of_fn_calls_to_start_job(steps_from_llm, tool_descriptions, fn_order):
    
    system_prompt = f"""You are given a set of steps to solve a problem and a bunch of tools that you can use to perform the steps
                        If you think a particular step cannot be solved using any of the tools, simply skip it.
                        The tools you have access to are:
                        {"".join(tool_descriptions)}
                        """
    
    agent_job = """Plan the sequence of function calls needed to execute the tasks given the steps.
                Return a JSON array of function calls in order that needs to be executed.
                Don't give any arguments in your sequence of function calls.Just return one json file with function names and steps following the template below. 
                [{'step': 'step_number', 'function': 'function_name'}].
                """
    task_identifier_agent = Agent(system_prompt=system_prompt + agent_job)
    response = task_identifier_agent.perform_action(steps_from_llm)

    return response

def function_call_validator(function_calls, fn_order):

    system_prompt = """You are given a  list of function calls that are helpful in solving a particular task.
                        Your job is to validate whether the sequence are in the correct order or not.
                        To validate you'd be given the tool sequence to check if the given function calls
                        are following the tool sequence.
                        Tools Sequence:
                        {}
                        """.format(fn_order)
    
    agent_job = """Check if function calls follow the order or not.
                    If they don't, align them correctly.
                    Return only the ordered function calls with the below template.

                    "'function_calls': [{'step': 'step_number', 'function': 'function_name'}]".
                    """
    
    validator_agent = Agent(system_prompt=system_prompt + agent_job)
    response = validator_agent.perform_action(function_calls)

    new_job= f"""Ensure that the below response is a valid JSON.
                {response}
                
                Do not say that "HERE is your JSON", return only the valid JSON
                """
    second_validator_agent = Agent(system_prompt="""You are a JSON Validator""")
    final_response = second_validator_agent.perform_action(new_job)

    return final_response


def scheduler(user_query, folder_path):

    tools, desc = accumulate_tools()
    dict_info ={}
    fn_order = list(tools.keys())
    logger.info("Using the solver agent to break the problem into sub-problems\n")
    llm_response = get_list_of_steps_to_perform_user_query(user_query)
    function_calls = get_list_of_fn_calls_to_start_job(llm_response, desc, fn_order)
    validated_function_calls = function_call_validator(function_calls=function_calls, fn_order=fn_order)
    logger.info(validated_function_calls)
    logger.info(dict_info)
    if isinstance(dict_info, list):
        logger.info("Started scheduling the sub-tasks and tools......")
        logger.info("""
                ░░░░
                ░    ░
            {○_○}   ░ Processing...
            <|   |> ░
            |   |  ░
            ════   ░
                ░░░░
            """)
        
        for step in dict_info:
            func = tools.get(step["function"])
            logger.info(func)
            if func is not None:
                if step["function"] == "move_files_to_categories":
                    dest_map = func(source_dir=folder_path)
                    if dest_map:
                        logger.info("Successfully identified different categories of files and moved them to appropriate subfolders")
                        logger.info("Task Completed")
                    
                elif step["function"] == "compress_pdf":
                    logger.info("came here")
                    fname = ""
                    for file in dest_map:
                        if file.endswith(".pdf"):
                            fname = dest_map[file]
                            break
                    logger.info(fname)
                    if fname != "":
                        func(file_path=fname)
                        logger.info("Successfully used online services to compress the input pdf and saved the results.")
                        logger.info("Task is completed successfully!!!")   
                
                elif step["function"] == "compress_image":
                    fname = ""
                    for file in dest_map:
                        if file.endswith(".png"):
                            fname = dest_map[file]
                            break
                    if fname != "":
                        func(file_path=fname)
                        logger.info("Successfully used online services to compress the input pdf and saved the results.")
                        logger.info("Task is completed successfully!!!")   
                
                elif step["function"] == "process_todo_file":
                    func(folder_path=folder_path)
                
                else:
                    logger.info("There is no such available tool. Sorry couldn't schedule sub-task!!!")
                

        return "The user-query is resolved and the sub-tasks are completed!!!"
                     

    else:
        return "LLM was unable to fetch the tools required to do your job. Sorry for the inconvenience"
    


if __name__ == "__main__":
    user_quer = ""
    folder_path = ""
    resp =scheduler(user_quer, folder_path)
    print(resp)

