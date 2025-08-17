import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from agents.chat_agent import ChatAgent
import logging
import requests
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Debug: Confirm script is running
print("Starting flask_app.py execution")

# Initialize Flask app
print("Initializing Flask app")
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'your_secret_key')
CORS(app, resources={r"/*": {"origins": "[invalid url, do not cite] supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="[invalid url, do not cite]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print("Logging configured")

# Initialize ChatAgent instance
print("Initializing ChatAgent")
try:
    agent = ChatAgent()
    logger.info("ChatAgent initialized successfully")
    print("ChatAgent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChatAgent: {str(e)}")
    print(f"Failed to initialize ChatAgent: {str(e)}")
    raise

# Initialize LangChain for AI responses
print("Initializing LangChain")
try:
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=os.environ.get("OPENAI_API_KEY", "your_openai_api_key"),
        temperature=0.7
    )
    logger.info("LangChain initialized successfully")
    print("LangChain initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LangChain: {str(e)}")
    print(f"Failed to initialize LangChain: {str(e)}")
    raise

# Define the /process route
@app.route('/process', methods=['POST'])
async def process():
    """
    Process incoming POST requests with input_type and input_data.
    Returns a JSON response with the result from ChatAgent or an error message.
    """
    try:
        data = request.get_json()
        logger.info(f"Received request data: {data}")
        print(f"Received request data: {data}")
        
        if not data or 'input_type' not in data or 'input_data' not in data:
            logger.warning("Invalid input data received")
            print("Invalid input data received")
            return jsonify({'error': 'Invalid input data'}), 400
        
        if not data['input_data'].strip():
            logger.warning("Empty input data received")
            print("Empty input data received")
            return jsonify({'error': 'Input data cannot be empty'}), 400
        
        result = await agent.process_input(data['input_type'], data['input_data'])
        logger.info(f"ChatAgent result: {result}")
        print(f"ChatAgent result: {result}")
        
        return jsonify({'result': result})
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

# Define the /cometchat-webhook route (for legacy compatibility)
@app.route('/cometchat-webhook', methods=['POST'])
async def cometchat_webhook():
    try:
        data = request.get_json()
        user_id = data["sender"]["uid"]
        message_text = data["data"]["text"]
        
        result = await process_message(user_id, message_text)
        
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}")
        print(f"Error in webhook: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    print("Client connected")
    emit('response', {'text': 'Connected to ixome.ai chatbot!'})

@socketio.on('message')
async def handle_message(data):
    try:
        user_id = data.get('user_id', 'anonymous')
        message_text = data.get('text', '')
        if not message_text.strip():
            emit('response', {'text': 'Please enter a message.'})
            return
        
        result = await process_message(user_id, message_text)
        emit('response', {'text': result})
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        print(f"Error handling message: {str(e)}")
        emit('response', {'text': f"Error: {str(e)}"})

async def process_message(user_id, message_text):
    # Classify message as technical or non-technical
    is_technical = classify_message(message_text)
    
    if is_technical:
        # Check subscription status via Strapi
        if not check_subscription(user_id):
            return "Please subscribe to our support plan for technical assistance."
    
    # Process with ChatAgent or LangChain
    try:
        result = await agent.process_input("text", message_text)
    except Exception:
        # Fallback to LangChain
        messages = [HumanMessage(content=message_text)]
        result = llm(messages).content
    
    return result

def classify_message(text):
    # Simple keyword-based classification
    technical_keywords = ["error", "bug", "crash", "install", "configure", "troubleshoot"]
    return any(keyword in text.lower() for keyword in technical_keywords)

def check_subscription(user_id):
    try:
        response = requests.get(f"[invalid url, do not cite])
        if response.status_code == 200:
            subscriptions = response.json()['data']
            if subscriptions:
                subscription = subscriptions[0]['attributes']
                tier = subscription['tier']
                problems_solved = subscription['problems_solved']
                if tier == "one_problem" and problems_solved < 1:
                    return True
                elif tier == "three_problems" and problems_solved < 3:
                    return True
                elif tier == "unlimited" and problems_solved < 100:
                    return True
        return False
    except Exception as e:
        logger.error(f"Error checking subscription: {str(e)}")
        print(f"Error checking subscription: {str(e)}")
        return False

# Wrap Flask app for ASGI compatibility
print("Wrapping Flask app with WsgiToAsgi")
asgi_app = SocketIO(app)

@app.route('/')
def home():
    return "Flask app is running!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    socketio.run(app, host='0.0.0.0', port=port)