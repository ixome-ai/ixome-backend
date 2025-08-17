import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging
import asyncio
from google.cloud import speech, vision
from langgraph.graph import Graph
from typing import Dict, Any, Optional
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone as PineconeClient
import cv2
import numpy as np
from dotenv import load_dotenv
import getpass
import time
from langchain_openai import OpenAIEmbeddings
from core.db import insert_dealer_info, query_sqlite

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import requests

# Load environment variables
load_dotenv(dotenv_path='/home/vincent/ixome/.env', override=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify and set API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in .env file!")
    openai_api_key = getpass.getpass("Please enter OPENAI_API_KEY: ")
    os.environ["OPENAI_API_KEY"] = openai_api_key

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    logger.error("PINECONE_API_KEY not found in .env file!")
    pinecone_api_key = getpass.getpass("Please enter PINECONE_API_KEY: ")
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

strapi_token = os.getenv("STRAPI_TOKEN")
if not strapi_token:
    logger.error("STRAPI_TOKEN not found in .env file!")
    strapi_token = getpass.getpass("Please enter STRAPI_TOKEN: ")
    os.environ["STRAPI_TOKEN"] = strapi_token

google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not google_credentials_path:
    logger.error("GOOGLE_APPLICATION_CREDENTIALS not found in .env file!")
    google_credentials_path = getpass.getpass("Please enter GOOGLE_APPLICATION_CREDENTIALS path: ")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path

# Initialize clients
client = OpenAI(api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
pc = PineconeClient(api_key=pinecone_api_key)
speech_client = speech.SpeechClient()
vision_client = vision.ImageAnnotatorClient()

# Initialize Pinecone index
index_name = "troubleshooter-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric='cosine',
        spec={'serverless': {'cloud': 'aws', 'region': os.getenv("PINECONE_ENVIRONMENT", "us-east-1")}}
    )
index = pc.Index(index_name)

# Flask-SocketIO setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Define Pydantic models
class ClientQuery(BaseModel):
    query: str
    client_id: Optional[str] = None
    timestamp: str

class Solution(BaseModel):
    solution: str
    confidence: float
    source: Optional[str] = None
    tokens_used: int = 0

class AgentState(BaseModel):
    input_type: Optional[str] = None
    input_data: Optional[Any] = None
    query: Optional[ClientQuery] = None
    processed_input: Optional[str] = None
    issue: Optional[str] = None
    solution: Optional[Solution] = None
    result: Dict = {}
    user_id: Optional[str] = None
    tokens_used: int = 0

class ChatAgent:
    def __init__(self):
        self.logger = logger
        self.client = client
        self.speech_client = speech_client
        self.vision_client = vision_client
        self.index = index

        self.graph = Graph()
        self.graph.add_node("input", self.input_node)
        self.graph.add_node("text_processing", self.text_processing_node)
        self.graph.add_node("voice_processing", self.voice_processing_node)
        self.graph.add_node("video_processing", self.video_processing_node)
        self.graph.add_node("issue_identification", self.issue_identification_node)
        self.graph.add_node("solution_retrieval", self.solution_retrieval_node)
        self.graph.add_node("response_generation", self.response_generation_node)

        self.graph.add_conditional_edges(
            "input",
            lambda state: state.input_type or "text",
            {"text": "text_processing", "voice": "voice_processing", "video": "video_processing"}
        )
        self.graph.add_edge("text_processing", "issue_identification")
        self.graph.add_edge("voice_processing", "issue_identification")
        self.graph.add_edge("video_processing", "issue_identification")
        self.graph.add_edge("issue_identification", "solution_retrieval")
        self.graph.add_edge("solution_retrieval", "response_generation")
        self.graph.set_entry_point("input")
        self.graph.set_finish_point("response_generation")
        self.app = self.graph.compile()

    async def input_node(self, state: AgentState) -> AgentState:
        self.logger.info(f"Received input: type={state.input_type}, data=<data>, user_id={state.user_id}")
        return state

    async def text_processing_node(self, state: AgentState) -> AgentState:
        if state.query and state.query.query:
            state.processed_input = state.query.query
            self.logger.info(f"Processed text input: {state.processed_input}")
        else:
            state.processed_input = ""
            self.logger.warning("No query provided for text processing")
        return state

    async def voice_processing_node(self, state: AgentState) -> AgentState:
        audio_data = state.input_data or b""
        if audio_data:
            try:
                audio = speech.RecognitionAudio(content=audio_data)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    language_code='en-US'
                )
                response = self.speech_client.recognize(config=config, audio=audio)
                if response.results:
                    state.processed_input = response.results[0].alternatives[0].transcript
                    self.logger.info(f"Processed voice input: {state.processed_input}")
                else:
                    state.processed_input = "No speech detected"
                    self.logger.info("No speech detected in audio")
            except Exception as e:
                self.logger.error(f"Error processing voice: {e}")
                state.processed_input = "Error processing voice"
        else:
            state.processed_input = "No audio data provided"
            self.logger.warning("No audio data provided for voice processing")
        return state

    async def video_processing_node(self, state: AgentState) -> AgentState:
        video_data = state.input_data or b""
        if video_data:
            try:
                temp_file = "temp_video.mp4"
                with open(temp_file, "wb") as f:
                    f.write(video_data)
                cap = cv2.VideoCapture(temp_file)
                frame_descriptions = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _, buffer = cv2.imencode('.jpg', frame)
                    image = vision.Image(content=buffer.tobytes())
                    response = self.vision_client.label_detection(image=image)
                    labels = [label.description for label in response.label_annotations]
                    frame_descriptions.append(", ".join(labels))
                cap.release()
                os.remove(temp_file)
                state.processed_input = "; ".join(frame_descriptions[:3]) if frame_descriptions else "No labels detected"
                self.logger.info(f"Processed video input: {state.processed_input}")
            except Exception as e:
                self.logger.error(f"Error processing video: {e}")
                state.processed_input = "Error processing video"
        else:
            state.processed_input = "No video data provided"
            self.logger.warning("No video data provided for video processing")
        return state

    async def issue_identification_node(self, state: AgentState) -> AgentState:
        processed_input = state.processed_input.lower() if state.processed_input else ""
        self.logger.info(f"Identifying issue from: {processed_input}")
        if any(phrase in processed_input for phrase in ["no sound", "sound not working", "surround sound", "audio issue"]):
            state.issue = "no_sound"
        elif "tv not turning on" in processed_input:
            state.issue = "tv_not_turning_on"
        elif "settings" in processed_input:
            state.issue = "settings_issue"
        elif any(phrase in processed_input for phrase in ["flashing light", "error code", "blinking"]):
            state.issue = "error_code"
        else:
            state.issue = "unknown"
        self.logger.info(f"Identified issue: {state.issue}")
        return state

    async def solution_retrieval_node(self, state: AgentState) -> AgentState:
        try:
            subscription = check_subscription(state.user_id)
            if subscription["status"] == "error" or not subscription.get("is_active", False):
                state.solution = Solution(solution="No active subscription. Please upgrade.", confidence=1.0, source="Strapi", tokens_used=0)
                return state
            if subscription["tokens_used"] >= subscription["tokens_limit"]:
                state.solution = Solution(solution="Token limit reached. Please upgrade your subscription.", confidence=1.0, source="Strapi", tokens_used=0)
                return state

            embedding_input = state.processed_input or "unknown issue"
            embedding = embeddings.embed_query(embedding_input)
            results = self.index.query(vector=embedding, top_k=1, include_metadata=True)
            tokens_used = len(embedding_input.split()) * 2
            if results['matches']:
                solution_text = results['matches'][0]['metadata'].get('solution', "No solution found")
                confidence = results['matches'][0]['score']
                sqlite_result = query_sqlite("Lutron" if "lutron" in embedding_input.lower() else "Unknown", state.issue or "Unknown")
                solution_text += f"\nExact dealer info from SQLite: {sqlite_result}"
                state.solution = Solution(solution=solution_text, confidence=confidence, source="Pinecone + SQLite", tokens_used=tokens_used)
            else:
                state.solution = Solution(solution="No solution found in Pinecone.", confidence=0.5, source="Pinecone", tokens_used=tokens_used)
            update_subscription(state.user_id, subscription["tokens_used"] + tokens_used)
        except Exception as e:
            self.logger.error(f"Pinecone/SQLite query failed: {e}")
            state.solution = Solution(solution=f"Error retrieving solution: {e}", confidence=0.0, source="Error", tokens_used=0)
        return state

    async def response_generation_node(self, state: AgentState) -> AgentState:
        if state.solution and state.solution.solution:
            response = f"Little Einstein: {state.solution.solution} (For user: {state.user_id})"
        else:
            response = f"Little Einstein: No specific solution found. Please check the device manual or contact support. (For user: {state.user_id})"
        state.result = {"status": "success", "response": response, "message": "Solution provided", "tokens_used": state.solution.tokens_used}
        self.logger.info(f"Generated response: {response}")
        return state

    async def process_input(self, input_type: str, input_data: Any, user_id: str) -> Dict[str, Any]:
        state = AgentState(
            input_type=input_type,
            input_data=input_data if input_type != "text" else None,
            query=ClientQuery(query=input_data if input_type == "text" else "", timestamp=time.strftime("%Y-%m-%d %H:%M:%S")),
            user_id=user_id
        )
        result = await self.app.ainvoke(state)
        return result.result

def check_subscription(username: str) -> dict:
    # Hardcode success for testing
    return {
        "status": "success",
        "username": username,
        "tier": "basic",
        "tokens_used": 0,
        "tokens_limit": 100,
        "is_active": True
    }

def update_subscription(username: str, tokens_used: int):
    try:
        response = requests.get(
            "http://127.0.0.1:1337/api/subscriber-ids",
            headers={"Authorization": f"Bearer {strapi_token}"},
            params={"filters[subscriber_id][$eq]": username}
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "data" in data and len(data["data"]) > 0:
            subscription_id = data["data"][0]["id"]
            response = requests.put(
                f"http://127.0.0.1:1337/api/subscriber-ids/{subscription_id}",
                headers={"Authorization": f"Bearer {strapi_token}"},
                json={"data": {"tokens_used": tokens_used}}
            )
            response.raise_for_status()
    except Exception as e:
        logger.error(f"Subscription update failed: {e}")

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    try:
        response = requests.post(
            "http://127.0.0.1:1337/api/auth/local",
            json={"identifier": f"{username}@ixome.ai", "password": password}
        )
        logger.info(f"Strapi auth response: {response.status_code}, {response.text}")
        response.raise_for_status()
        auth_data = response.json()
        if auth_data.get("jwt"):
            subscription = check_subscription(username)
            logger.info(f"Subscription check result: {subscription}")
            if subscription["status"] == "success" and subscription.get("is_active", False):
                return jsonify({'status': 'success', 'message': 'Logged in successfully', 'jwt': auth_data["jwt"]})
            return jsonify({'status': 'error', 'message': 'No active subscription'}), 403
        return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401
    except requests.exceptions.HTTPError as e:
        logger.error(f"Login failed: {e}, Response: {e.response.text if e.response else 'No response'}")
        return jsonify({'status': 'error', 'message': f'Login error: {e.response.text if e.response else str(e)}'}), 500
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return jsonify({'status': 'error', 'message': f'Login error: {str(e)}'}), 500

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connection_response', {'message': 'Connected to Socket.IO server'})

@socketio.on('message')
def handle_message(data):
    logger.info(f'Received message: {data}')
    message = data.get('message')
    user_id = data.get('user_id', 'unknown')
    non_technical_keywords = ["website", "navigation", "login", "signup", "how to use"]
    if any(keyword in message.lower() for keyword in non_technical_keywords):
        response = {"sender": "bot", "text": f"Little Einstein: For {message}, please visit our help page or contact support."}
        emit('message', response)
        return
    agent = ChatAgent()
    result = asyncio.run(agent.process_input("text", message, user_id))
    logger.info(f"Bot response: {result['response']}")
    emit('message', {
        'sender': 'bot',
        'text': result["response"],
        'tokens_used': result.get("tokens_used", 0)
    })

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5003, debug=True)