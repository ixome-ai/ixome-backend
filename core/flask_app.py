# Complete Flask app with homepage data route, marketing/social agents for dynamic boxes, fits existing setup.
import sys
import os
sys.path.append('/home/vincent/ixome')  # Absolute path to include agents directory

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_socketio import SocketIO, emit
from agents.chat_agent import ChatAgent
from agents.marketing_agent import MarketingAgent
from agents.social_agent import SocialAgent
import logging
from asgiref.wsgi import WsgiToAsgi
from dotenv import load_dotenv
import requests
import random

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', '1c5565f83a9180cd3b7c544da8d8faf1623613d1b6c50f06d6702d8f6d641779')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your_default_jwt_secret_key')

# Set up CORS
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://ixome.ai"]}}, supports_credentials=True)

# Initialize JWT and SocketIO
jwt = JWTManager(app)
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000", "https://ixome.ai"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize agents
print("Initializing agents")
try:
    agent = ChatAgent()
    marketing_agent = MarketingAgent()
    social_agent = SocialAgent()
    logger.info("Agents initialized successfully")
    print("Agents initialized successfully")
except ImportError as e:
    logger.error(f"Failed to import agents: {str(e)}")
    print(f"Failed to import agents: {str(e)}")
    raise
except Exception as e:
    logger.error(f"Failed to initialize agents: {str(e)}")
    print(f"Failed to initialize agents: {str(e)}")
    raise

# Strapi API configuration
STRAPI_URL = os.environ.get('STRAPI_URL', 'http://localhost:1337')

def check_subscription(user_id):
    try:
        response = requests.get(f"{STRAPI_URL}/api/users?filters[username][$eq]={user_id}", headers={'Authorization': 'Bearer your_strapi_jwt'})
        if response.status_code == 200:
            user = response.json().get('data', [{}])[0]
            if user:
                subscription = user.get('attributes', {}).get('subscription', {})
                tokens = subscription.get('tokens', 0)
                if tokens > 0 or (not user.get('attributes', {}).get('first_visit_done', False)):
                    return True
                return False
        return False
    except Exception as e:
        logger.error(f"Error checking subscription: {e}")
        return False

# Homepage data endpoint (for dynamic boxes from agents)
@app.route('/homepage_data')
def homepage_data():
    try:
        campaign = marketing_agent.generate_campaign()  # Returns {'content': 'Promo text...'}
        boxes = [
            {'title': 'Basic Tier', 'description': '100 tokens - $10 ' + campaign.get('content', ''), 'link': '/purchase/basic'},
            {'title': 'Pro Tier', 'description': '1000 tokens - $50 ' + campaign.get('content', ''), 'link': '/purchase/pro'},
            {'title': 'Enterprise Tier', 'description': '10000 tokens - $200 ' + campaign.get('content', ''), 'link': '/purchase/enterprise'}
        ]
        # Agentic: Trigger social promotion if random (or CEO logic for low subscriptions)
        if random.random() < 0.2:
            social_agent.promote()
        logger.info(f"Generated homepage data: {boxes}")
        return jsonify(boxes=boxes)
    except Exception as e:
        logger.error(f"Error generating homepage data: {e}")
        return jsonify({"error": "Failed to load homepage data"}), 500

# Socket.IO event handlers
@socketio.on('connect')
@jwt_required()
def handle_connect():
    current_user = get_jwt_identity()
    logger.info(f"User {current_user} connected")
    emit('response', {'text': f"Hey {current_user}! Welcome to ixome.ai chatbot! How can I help you today?"})

@socketio.on('message')
@jwt_required()
async def handle_message(data):
    current_user = get_jwt_identity()
    user_message = data.get('text', '').strip()
    if not user_message:
        emit('response', {'text': 'Oops! Please type a message to get started!'})
        return

    logger.info(f"User {current_user} sent: {user_message}")
    is_technical = any(keyword in user_message.lower() for keyword in ["error", "bug", "crash", "install", "configure"])

    if is_technical:
        if not check_subscription(current_user):
            emit('response', {
                'text': "This looks like a technical issue! I can solve one easy problem for free. If it’s complex, please subscribe to one of our plans: $10 (1 problem), $50 (10 problems), or $200 (100 problems). Visit /support to subscribe!",
                'redirect': '/support'
            })
            return

    try:
        result = await agent.process_input("text", user_message, current_user)
        emit('response', {'text': result})
        if is_technical and check_subscription(current_user):
            tokens = requests.get(f"{STRAPI_URL}/api/users?filters[username][$eq]={current_user}", headers={'Authorization': 'Bearer your_strapi_jwt'}).json().get('data', [{}])[0].get('attributes', {}).get('subscription', {}).get('tokens', 0) - 1
            requests.put(f"{STRAPI_URL}/api/users/{current_user}", json={'tokens': tokens}, headers={'Authorization': 'Bearer your_strapi_jwt'})
            emit('response', {'text': "Follow-up: Need more help? Ask another question or let me know!"})
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        emit('response', {'text': f"Oops! Something went wrong. Try again later! ({str(e)})"})

# Define the /login route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    # TODO: Replace with proper authentication (e.g., Strapi or database)
    if username == 'test' and password == 'test':
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token), 200
    return jsonify({"msg": "Bad credentials"}), 401

# Define the /process route for API access
@app.route('/process', methods=['POST'])
@jwt_required()
async def process():
    try:
        data = request.get_json()
        logger.info(f"Received request data: {data}")
        if not data or 'input_type' not in data or 'input_data' not in data:
            return jsonify({'error': 'Invalid input data'}), 400
        if not data['input_data'].strip():
            return jsonify({'error': 'Input data cannot be empty'}), 400
        current_user = get_jwt_identity()
        logger.info(f"Processing request for user: {current_user}")
        result = await agent.process_input(data['input_type'], data['input_data'], current_user)
        logger.info(f"ChatAgent result: {result}")
        return jsonify({'result': result})
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/')
def home():
    return "Flask app is running!"

# Wrap Flask app for ASGI compatibility
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)