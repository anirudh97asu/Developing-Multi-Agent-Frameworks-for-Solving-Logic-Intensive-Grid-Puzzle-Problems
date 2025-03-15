import warnings
import os
import json

from google import genai
from tqdm import tqdm
from dotenv import load_dotenv
from google import generativeai
from google.genai import types
from google.generativeai.types import GenerationConfig #GenerateContentConfig
warnings.filterwarnings("ignore")

load_dotenv()


client = genai.Client(api_key=os.getenv("API_KEY"))

class Agent:

    def __init__(self, system_prompt=None):
        self.model_name = "gemini-2.0-flash-exp"
        self.system_prompt = system_prompt
        self.client = client

        self.chat = self.client.chats.create(  model=self.model_name,
                                              
                                                config=types.GenerateContentConfig(system_instruction=self.system_prompt,
                                                                                   temperature=0.0,)
                                            )

    def perform_action(self, user_query):

        response = self.chat.send_message(user_query)
        return response.text    