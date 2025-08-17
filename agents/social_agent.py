from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os
import tweepy
import logging

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path='/home/vincent/ixome/.env')

class SocialAgent:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.llm = OpenAI(temperature=0.7, openai_api_key=api_key)
        self.prompt_template = PromptTemplate(
            input_variables=["topic"],
            template="Generate a concise, engaging X post (280 characters or less) for IXome.ai on the topic '{topic}' with SEO keywords 'smart home chatbot', 'Control4 troubleshooting', 'Lutron support'."
        )
        self.chain = self.prompt_template | self.llm
        # Initialize X API
        consumer_key = os.environ.get("X_CONSUMER_KEY")
        consumer_secret = os.environ.get("X_CONSUMER_SECRET")
        access_token = os.environ.get("X_ACCESS_TOKEN")
        access_token_secret = os.environ.get("X_ACCESS_TOKEN_SECRET")
        if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
            logger.warning("X API credentials not found; social posts will not be sent")
            self.client = None
        else:
            self.client = tweepy.Client(
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token=access_token,
                access_token_secret=access_token_secret
            )

    def promote(self, topic="smart home automation"):
        """Generate and post to X."""
        try:
            post_content = self.chain.invoke({"topic": topic}).strip()
            if self.client:
                self.client.create_tweet(text=post_content)
                logger.info(f"Posted to X: {post_content}")
            else:
                logger.info(f"Generated X post (not sent, no credentials): {post_content}")
            return {"status": "success", "post": post_content}
        except Exception as e:
            logger.error(f"Error generating or posting to X: {e}")
            return {"status": "error", "message": str(e)}