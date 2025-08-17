def optimize_content(details: Dict) -> Dict:
    keyword = details.get("keyword", "default_keyword")
    content = details.get("content", "Sample content to optimize")
    prompt = f"Optimize this content for SEO around '{keyword}':\n{content}"
    try:
        response = client.chat.completions.create(
            model="grok-2-1212",  # Double-check this model name in xAI's docs
            messages=[{"role": "user", "content": prompt}]
        )
        # Log the raw response for debugging
        print(f"Raw API Response: {response}")
        
        # Check if response is None
        if response is None:
            print("Debug: Response is None")
            return {"status": "error", "message": "API response is None"}
        
        # Check if 'choices' exists and is a list with at least one item
        if not hasattr(response, 'choices'):
            print("Debug: Response has no 'choices' attribute")
            return {"status": "error", "message": "No 'choices' attribute in API response"}
        if not response.choices:
            print("Debug: Choices is empty or None")
            return {"status": "error", "message": "Choices is empty or None in API response"}
        
        # Access the first choice and check for 'message'
        choice = response.choices[0]
        if not hasattr(choice, 'message'):
            print("Debug: Choice has no 'message' attribute")
            return {"status": "error", "message": "No 'message' attribute in API response choice"}
        if not choice.message:
            print("Debug: Message is None")
            return {"status": "error", "message": "Message is None in API response choice"}
        
        # Access the content
        optimized_content = choice.message.content
        if optimized_content is None:
            print("Debug: Content is None")
            return {"status": "error", "message": "Content is None in API response"}
        
        return {"status": "success", "optimized_content": optimized_content, "message": "Content optimized"}
    
    except Exception as e:
        # Log the exception details
        print(f"Exception occurred: {type(e).__name__}: {str(e)}")
        return {"status": "error", "message": f"Failed to optimize content: {str(e)}"}