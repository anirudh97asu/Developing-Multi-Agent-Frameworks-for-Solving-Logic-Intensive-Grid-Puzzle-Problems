import requests
import json
from ollama import chat

class ChatAgent:
    def __init__(self, model_name, system_prompt="You are a helpful assistant."):
        """
        Initialize a ChatAgent with a specific model and system prompt.
        
        Args:
            model_name (str): The name of the model to use
            system_prompt (str): The system prompt to use for all conversations
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.conversation_history = []
        # Initialize with system message
        self.conversation_history.append({"role": "system", "content": self.system_prompt})
    
    def chat(self, user_message):
        """
        Send a message to the model and get a response.
        
        Args:
            user_message (str): The user's message
            
        Returns:
            str: The model's response
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Make the API call
        response = self._call_chat_api()
        
        # Extract assistant message and add to history
        assistant_message = response["message"]["content"]
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def _call_chat_api(self):
        """
        Call the chat API with the current conversation history.
        
        Returns:
            dict: The raw API response
        """
        # This is a placeholder for your actual API call implementation
        # Replace with your actual API call code
        return chat(
            model=self.model_name,
            messages=self.conversation_history
        )
    
    def reset_conversation(self):
        """
        Reset the conversation history, keeping only the system prompt.
        """
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
    
    def get_conversation_history(self):
        """
        Get the full conversation history.
        
        Returns:
            list: The conversation history
        """
        return self.conversation_history

# Example usage
if __name__ == "__main__":
    # Assuming the chat function is defined elsewhere
    
    # Create a Deepseek agent
    agent = ChatAgent(model_name="your_model_name")
    
    # Have a conversation
    response = agent.chat("What is the capital of France?")
    print(f"Agent: {response}")
    
    response = agent.chat("What about Germany?")
    print(f"Agent: {response}")
    
    # Get conversation history
    history = agent.get_conversation_history()
    print("\nConversation history:")
    for message in history:
        print(f"{message['role']}: {message['content']}")