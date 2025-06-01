# api_alternatives.py
"""
Alternative APIs for text summarization with cost comparison
"""

import openai
import anthropic
import requests
import time
from abc import ABC, abstractmethod

class SummarizerAPI(ABC):
    """Abstract base class for summarization APIs"""
    
    @abstractmethod
    def summarize_text(self, text: str, chapter_title: str = "") -> str:
        pass
    
    @abstractmethod
    def get_cost_per_1k_tokens(self) -> dict:
        pass

class GeminiSummarizer(SummarizerAPI):
    """
    Google Gemini API
    Pros: Very cheap, good quality, large context window
    Cons: Rate limits, newer API
    Cost: ~$0.00015 per 1K input tokens, $0.0006 per 1K output tokens
    """
    
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def summarize_text(self, text: str, chapter_title: str = "") -> str:
        prompt = f"""
        Summarize the following chapter content comprehensively:
        Chapter: {chapter_title}
        
        Content: {text}
        
        Provide a detailed summary (200-400 words) covering main ideas, key points, and important details.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error with Gemini API: {str(e)}"
    
    def get_cost_per_1k_tokens(self) -> dict:
        return {"input": 0.00015, "output": 0.0006}

class OpenAISummarizer(SummarizerAPI):
    """
    OpenAI GPT API
    Pros: High quality, reliable, well-documented
    Cons: More expensive than Gemini
    Cost: GPT-3.5-turbo ~$0.001 per 1K tokens, GPT-4 ~$0.03 per 1K tokens
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
    
    def summarize_text(self, text: str, chapter_title: str = "") -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating comprehensive chapter summaries."},
                    {"role": "user", "content": f"""
                    Create a detailed summary of this chapter:
                    Chapter: {chapter_title}
                    
                    Content: {text}
                    
                    Requirements:
                    - 200-400 words
                    - Cover main ideas and key points
                    - Include important details and examples
                    - Maintain logical flow
                    """}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"
    
    def get_cost_per_1k_tokens(self) -> dict:
        if "gpt-4" in self.model:
            return {"input": 0.03, "output": 0.06}
        else:  # gpt-3.5-turbo
            return {"input": 0.001, "output": 0.002}

class ClaudeSummarizer(SummarizerAPI):
    """
    Anthropic Claude API
    Pros: High quality, good at following instructions, large context
    Cons: More expensive than Gemini, requires different API structure
    Cost: ~$0.008 per 1K input tokens, $0.024 per 1K output tokens
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def summarize_text(self, text: str, chapter_title: str = "") -> str:
        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Cheapest Claude model
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"""
                    Create a comprehensive summary of this chapter:
                    Chapter Title: {chapter_title}
                    
                    Content: {text}
                    
                    Requirements:
                    - 200-400 words
                    - Cover all main ideas and key points
                    - Include important details and examples
                    - Use clear, readable language
                    """
                }]
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"Error with Claude API: {str(e)}"
    
    def get_cost_per_1k_tokens(self) -> dict:
        return {"input": 0.00025, "output": 0.00125}  # Haiku pricing

class HuggingFaceSummarizer(SummarizerAPI):
    """
    Hugging Face Transformers (Free/Self-hosted)
    Pros: Free, privacy-friendly, can run offline
    Cons: Requires more setup, may need powerful hardware
    Cost: Free (but requires compute resources)
    """
    
    def __init__(self):
        try:
            from transformers import pipeline
            self.summarizer = pipeline("summarization", 
                                     model="facebook/bart-large-cnn",
                                     device=-1)  # Use CPU
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")
    
    def summarize_text(self, text: str, chapter_title: str = "") -> str:
        try:
            # Split text if too long (BART has token limits)
            max_length = 1024
            if len(text) > max_length:
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                summaries = []
                for chunk in chunks:
                    result = self.summarizer(chunk, max_length=150, min_length=50)
                    summaries.append(result[0]['summary_text'])
                return ' '.join(summaries)
            else:
                result = self.summarizer(text, max_length=200, min_length=100)
                return result[0]['summary_text']
        except Exception as e:
            return f"Error with Hugging Face model: {str(e)}"
    
    def get_cost_per_1k_tokens(self) -> dict:
        return {"input": 0.0, "output": 0.0}  # Free but requires compute

class ConfigurableSummarizer:
    """Main class that can use any of the above APIs"""
    
    def __init__(self, api_type: str, api_key: str = None, **kwargs):
        """
        Initialize with chosen API
        
        Args:
            api_type: "gemini", "openai", "claude", or "huggingface"
            api_key: API key (not needed for huggingface)
            **kwargs: Additional arguments (e.g., model name for OpenAI)
        """
        
        if api_type == "gemini":
            self.api = GeminiSummarizer(api_key)
        elif api_type == "openai":
            model = kwargs.get("model", "gpt-3.5-turbo")
            self.api = OpenAISummarizer(api_key, model)
        elif api_type == "claude":
            self.api = ClaudeSummarizer(api_key)
        elif api_type == "huggingface":
            self.api = HuggingFaceSummarizer()
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
        
        self.api_type = api_type
    
    def summarize_text(self, text: str, chapter_title: str = "") -> str:
        """Summarize text using the configured API"""
        return self.api.summarize_text(text, chapter_title)
    
    def get_cost_estimate(self, total_tokens: int) -> dict:
        """Get estimated cost for processing"""
        costs = self.api.get_cost_per_1k_tokens()
        estimated_cost = {
            "input_cost": (total_tokens / 1000) * costs["input"],
            "output_cost": (total_tokens / 1000) * costs["output"] * 0.3,  # Assume output is 30% of input
            "total_cost": ((total_tokens / 1000) * costs["input"]) + ((total_tokens / 1000) * costs["output"] * 0.3)
        }
        return estimated_cost

# Cost comparison for a typical book (100,000 tokens)
def compare_api_costs():
    """Compare costs across different APIs for a typical book"""
    
    apis = {
        "Gemini Flash": {"input": 0.00015, "output": 0.0006},
        "GPT-3.5-turbo": {"input": 0.001, "output": 0.002},
        "GPT-4": {"input": 0.03, "output": 0.06},
        "Claude Haiku": {"input": 0.00025, "output": 0.00125},
        "Claude Sonnet": {"input": 0.003, "output": 0.015},
        "Hugging Face": {"input": 0.0, "output": 0.0}
    }
    
    tokens = 100000  # Typical book size
    output_ratio = 0.3  # Output is typically 30% of input
    
    print("Cost Comparison for Processing 100K Token Book:")
    print("=" * 50)
    
    for api_name, costs in apis.items():
        input_cost = (tokens / 1000) * costs["input"]
        output_cost = (tokens * output_ratio / 1000) * costs["output"]
        total_cost = input_cost + output_cost
        
        print(f"{api_name:15} | ${total_cost:.4f}")
    
    print("\nRecommendations:")
    print("• Gemini Flash: Best balance of cost and quality")
    print("• GPT-3.5-turbo: Good quality, moderate cost")
    print("• Hugging Face: Free but requires your own compute")
    print("• Claude Haiku: Good quality, reasonable cost")

# Example usage
if __name__ == "__main__":
    compare_api_costs()
