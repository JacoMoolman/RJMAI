from openai import OpenAI
import os
import json

# Set up the OpenAI client with direct API key
client = OpenAI(
    api_key="YOUR_API_KEY_HERE",
)

def deep_research(query, max_completion_tokens=4000):
    """
    Perform deep research using OpenAI's o3 model.
    
    Args:
        query (str): The research question or topic
        max_completion_tokens (int): Maximum number of tokens in the response
        
    Returns:
        str: The research response from the model
    """
    print(f"Sending query: {query}")
    print(f"Using max_completion_tokens: {max_completion_tokens}")
    
    try:
        # Use the current client-based API
        response = client.chat.completions.create(
            model="o3-2025-04-16",
            messages=[
                {"role": "system", "content": "You are a research assistant. Provide thorough, accurate, and detailed information. Use Deep Research to get the best results."},
                {"role": "user", "content": query}
            ],
            max_completion_tokens=max_completion_tokens
        )
        
        # Print the full response for debugging
        print("Raw API response:")
        print(json.dumps(response.model_dump(), indent=2))
        
        content = response.choices[0].message.content
        print(f"Content length: {len(content) if content else 0}")
        
        # Check if content is empty and try to extract useful information from the response
        if not content:
            print("Content is empty, trying to extract useful information from the response...")
            # Try to check if there's useful data in the response
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                content = f"Tool calls: {response.choices[0].message.tool_calls}"
            elif hasattr(response.choices[0].message, 'function_call') and response.choices[0].message.function_call:
                content = f"Function call: {response.choices[0].message.function_call}"
            else:
                content = "The model returned an empty response. You might need to try a different query or check your API configuration."
        
        return content
    except Exception as e:
        print(f"Exception details: {type(e).__name__}: {e}")
        return f"Error in API call: {str(e)}"

# Example usage
if __name__ == "__main__":
    research_query = "Do an analisis of the EURUSD currency pair and based on that provide advice if one should BUY SELL or DO NOTHING"
    print("Starting deep research...")
    result = deep_research(research_query)
    print("\nRESULT:")
    print(result)
