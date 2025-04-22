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


    if agent_name == "llm_agent":
        
        class Recipe(BaseModel):
            ingredients: List[str]

        return Recipe 



    if agent_name == "prob_agent_1":

        class CustomFieldModel(BaseModel):
            Item_Item_Pair: str
            Probability: float = Field(ge=0.05, le=0.95, description="Float between 0 and 1")


        class Recipe(BaseModel):
            ingredients: List[CustomFieldModel]  # or List[Tuple[str, float, str]]


        return Recipe


    elif agent_name == "prob_agent_2":

        class Recipe(BaseModel):
            tool_name: str
            Topic_Pair: str
            Row_Index: int
            Col_Index: int
            Probability: float

        return Recipe
    
    else:
        return None


class Agent:

    def __init__(self,agent_name, system_prompt=None, json_schema=None):
        self.model_name = "gemini-2.0-flash-exp"
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.client = client

        if not json_schema:

            self.chat = self.client.chats.create(  model=self.model_name,
                                              
                                                config=types.GenerateContentConfig(system_instruction=self.system_prompt,
                                                                                   temperature=0.0,)
                                            )
            
        else:
            if json_schema is not None:
                print("I think there is some json schema", json_schema)
            self.chat = self.client.chats.create(model=self.model_name,
                                                config=types.GenerateContentConfig(system_instruction=self.system_prompt,
                                                                                    temperature=0.0,
                                                                                    top_k=18,
                                                                                    top_p = 0.4,
                                                                                    response_mime_type='application/json',
                                                                                    response_schema=json_schema))
            

    def perform_action(self, user_query):
        response = self.chat.send_message(user_query)
        return response.text    