from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='/home/vincent/ixome/.env')

class WritingAgent:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.llm = OpenAI(temperature=0.7, openai_api_key=api_key)
        self.prompt_template = PromptTemplate(
            input_variables=["topic"],
            template="Generate a detailed, engaging blog post on the topic: {topic}. Make it SEO-friendly with keywords like 'smart home chatbot', 'Control4 troubleshooting', 'Lutron support'. Structure it with an introduction, 3-5 main sections, and a conclusion calling to action for IXome.ai subscriptions."
        )
        self.chain = self.prompt_template | self.llm

    def generate_content(self, topic):
        try:
            return self.chain.invoke({"topic": topic}).strip()
        except Exception as e:
            return f"Error generating content: {str(e)}"