import warnings
import os
import json

from google import genai
from tqdm import tqdm
from dotenv import load_dotenv
from google import generativeai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Tuple, TypedDict


from google.generativeai.types import GenerationConfig #GenerateContentConfig
warnings.filterwarnings("ignore")

load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY"))


def get_json_schema(agent_name):


    if agent_name == "prob_agent_1":

        class CustomFieldModel(BaseModel):
            Topic_Pair: str
            Pair_Index: int = Field(ge=0, le=15, description="Integer between 0 and 16")
            Probability: float = Field(ge=0.0, le=1.0, description="Float between 0 and 1")


        class Recipe(BaseModel):
            ingredients: List[CustomFieldModel]  # or List[Tuple[str, float, str]]


        return Recipe


    elif agent_name == "prob_agent_2":

        class Recipe(BaseModel):
            tool_name: str
            matrix_name: str
            i: int
            j: int
            value: float

        return Recipe
    
    else:
        return None


class Agent:

    def __init__(self,agent_name, system_prompt=None, json=None):
        self.model_name = "gemini-2.0-flash-exp"
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.client = client

        if not json:

            self.chat = self.client.chats.create(  model=self.model_name,
                                              
                                                config=types.GenerateContentConfig(system_instruction=self.system_prompt,
                                                                                   temperature=0.0,)
                                            )
            
        else:
            recipe = get_json_schema(agent_name)

            if recipe is not None:
                self.chat = self.client.chats.create(model=self.model_name,
                                                    config=types.GenerateContentConfig(system_instruction=self.system_prompt,
                                                                                        temperature=0.0,
                                                                                        response_mime_type='application/json',
                                                                                        response_schema=list[recipe]))
                

    def perform_action(self, user_query):
        response = self.chat.send_message(user_query)
        return response.text    