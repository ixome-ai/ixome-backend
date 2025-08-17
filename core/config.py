import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file for local development
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'), override=True)

# Pinecone API key
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Google credentials
GOOGLE_CREDENTIALS_RAW = os.getenv('GOOGLE_CREDENTIALS_RAW')
if GOOGLE_CREDENTIALS_RAW:
    # Write credentials to a temporary file on Heroku
    GOOGLE_CREDENTIALS_PATH = '/app/credentials.json'
    try:
        with open(GOOGLE_CREDENTIALS_PATH, 'w') as f:
            f.write(GOOGLE_CREDENTIALS_RAW)
        logger.info("Successfully wrote GOOGLE_CREDENTIALS_RAW to /app/credentials.json")
    except Exception as e:
        logger.error(f"Failed to write credentials file: {e}")
        raise ValueError(f"Failed to write credentials file: {e}")
else:
    # Fallback for local development
    GOOGLE_CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), 'credentials.json')

# Validate credentials file
if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
    logger.error(f"Google credentials file not found at {GOOGLE_CREDENTIALS_PATH}")
    raise ValueError(f"Google credentials file not found at {GOOGLE_CREDENTIALS_PATH}")

# Validate Pinecone API key
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY environment variable is not set")
    raise ValueError("PINECONE_API_KEY environment variable is not set")

logger.info("Environment variables loaded successfully")