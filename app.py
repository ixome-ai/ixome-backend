# Add to /home/vincent/ixome/website/app.py (your existing Flask app)
from agents.marketing_agent import MarketingAgent
from agents.social_agent import SocialAgent
import random

marketing_agent = MarketingAgent()
social_agent = SocialAgent()

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