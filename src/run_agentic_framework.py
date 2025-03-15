import os
import logging

import traceback
from tqdm import tqdm
from src.llm_engine.gemini_agent import Agent
from src.llm_engine.scheduler import scheduler

__name__ = "__run_agentic_framework__"
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

def probability_agent(user_query):

    # system_prompt = """
    #                     Create probability matrices for each pair of user-provided categories, items, and clues. Each matrix maps one category’s values on the x-axis and another on the y-axis.
    #                     Process clues sequentially, updating probabilities while normalizing affected rows and columns. Maintain dependencies between clues to reflect cumulative impacts.
    #                     Ensure each probability matrix remains updated, showing relationships between subcategories across matrices. Repeat until all clues are processed, then return the final normalized matrices.
    #                     Your goal is to demonstrate probability calculations, not solve the puzzle. Finally print the updated matrices and never hint at the final solution.
    #                 """

    #      6. Maintain rules of probability by ensuring all row-vectors and column-vectors sum to 1 at each update.
    # system_prompt = """
    #                 You are an AI assistant specialized in creating probability matrices from user-provided clues and categories. Your role is to:

    #                 1. Generate a complete set of probability matrices - one for each pair of categories provided.
    #                 2. Structure each matrix with one category's values on the x-axis and another category's values on the y-axis.
    #                 3. Calculate and display probability relationships between subcategories across matrices.
    #                 4. Process clues sequentially step by step and update the required probability table.
    #                 5. Identify hidden implicit information between multiple clues after processing some clues and cumulatively update the tables.
    #                 7. Provide clear explanations of your reasoning process when updating probabilities at each step.

    #                 Your end goal is to demonstrate probability calculations, not solve the puzzle. Finally print the updated matrices.
    #                 """

    # 6. Identify and incorporate implicit information that emerges from the combination of multiple clues.
    # 7. After processing several clues, look for hidden logical connections between matrices and update accordingly.
    # 8. Provide clear explanations of your probability calculations and logical deductions at each step.
    # 9. Display updated matrices after each significant change in probabilities.
    
    system_prompt = """You are an AI assistant specialized in creating and managing probability matrices for logical puzzles. Your task is to:

                        1. Create probability matrices for each pair of categories provided in a puzzle (e.g., if given categories like people, colors, and numbers, you'll create people-colors, people-numbers, and colors-numbers matrices)
                        2. Format each matrix clearly with one category on the x-axis and another on the y-axis.
                        3. Process each clue methodically, showing your step-by-step reasoning for how each clue affects the probabilities in relevant matrices.
                        4. Update probabilities after each clue (1 for confirmed matches, 0 for impossibilities, and fractional values for partial information).
                        5. Normalize each affected row and column while preserving these constraints
                        6. Show your normalization calculations, explaining how uncertain probabilities are adjusted
                        7. Verify each row and column sums to 1 after normalization
                        Focus on demonstrating your probability calculation process rather than simply solving the puzzle. Your final output should include all completed probability matrices with your reasoning clearly documented."""

    task_identifier_agent = Agent(system_prompt=system_prompt)

    response = task_identifier_agent.perform_action(user_query=user_query)

    return response



def validator_agent(user_query):

    system_prompt = """You are an AI assistant specialized in verifying logical reasoning chains and probability calculations produced by another agent. Your task is to meticulously analyze each step of the provided solution for a logic puzzle or probability problem, focusing on:

            1. VALIDITY CHECK: Examine each reasoning step for logical consistency and correctness.
            - Identify any logical fallacies or incorrect deductions
            - Verify that all stated premises are used appropriately
            - Ensure no contradictory conclusions are drawn

            2. PROBABILITY VERIFICATION: Thoroughly validate all probability tables and calculations.
            - Confirm that probability distributions of rows and colums of the tables sum to 1.0 where appropriate

            3. COMPLETENESS ASSESSMENT: Determine if the solution addresses all aspects of the problem.
            - Verify all clues from the original problem are incorporated
            - Confirm all required matches between items and categories are established
            - Check that ambiguous cases are properly identified and resolved

            After validation, present the complete, corrected reasoning chain with all necessary probability calculations. Format your response with clear section headings and numbered steps to ensure the solution is easily followed."""


    validator_agent = Agent(system_prompt=system_prompt)

    response = validator_agent.perform_action(user_query=user_query)

    return response


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

    Hi! I am your AI assistant.. I can organize folders, compress files and send invites
    How may I help you today?

        
    """)

    try:

        pass

    
    except Exception as e:
        logger.info(f"There is some issue in running your process: {str(traceback.format_exc(e))}")

        
