
from dotenv import load_dotenv  # Add this line
load_dotenv()  # Add this line to load .env

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

class MarketingAgent:
    def __init__(self):
        self.llm = OpenAI(temperature=0.7)

        self.prompt = PromptTemplate.from_template(
            "You are a marketing agent for IXome.ai, a smart home chatbot subscription service. Generate a promotional campaign for {topic}.\n\n{tools}\n\nUse the following format:\n\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nThought:{agent_scratchpad}"
        )

        self.tools = [
            Tool(
                name="Generate Promo",
                func=self.generate_promo,
                description="Generate a promotional message"
            )
        ]

        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def generate_promo(self, input):
        return f"Special promo: {input} - Get 20% off your first subscription!"

    def generate_campaign(self, topic="subscription tiers"):
        result = self.agent_executor.invoke({"topic": topic, "agent_scratchpad": ""})
        return {'content': result['output']}

# For testing
if __name__ == "__main__":
    agent = MarketingAgent()
    print(agent.generate_campaign())