import pytest
import requests
from playwright.sync_api import Page, expect
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GeminiModel
from deepeval import assert_test

def test_ecommerce_chatbot_flow(page: Page):
    # ---------------------------------------------------------
    # LAYER 1: API (Fast Setup / Data Verification)
    # ---------------------------------------------------------
    # Check if the backend service is up before running the UI
    api_response = requests.get("https://dummyjson.com/products/1")
    assert api_response.status_code == 200, "Backend API is down!"
    product_name = api_response.json()["title"]

    # ---------------------------------------------------------
    # LAYER 2: UI (Playwright Deterministic Testing)
    # ---------------------------------------------------------
    # Navigate to our app and interact with the chatbot UI
    page.goto("https://www.example.com/support")
    
    # Simulate a user asking the chatbot a question
    user_query = f"Do you have the {product_name} in stock?"
    page.fill("#chat-input", user_query)
    page.click("#send-btn")
    
    # Wait for the chatbot to reply and extract its text
    page.wait_for_selector(".bot-message")
    actual_bot_response = page.inner_text(".bot-message")
    
    # Standard UI assertion to ensure the chat window didn't crash
    expect(page.locator(".bot-message")).to_be_visible()

    # ---------------------------------------------------------
    # LAYER 3: AI (DeepEval Probabilistic Testing with Gemini)
    # ---------------------------------------------------------
    # Initialize the free Gemini model for the evaluator [1]
    # DeepEval will automatically look for the GOOGLE_API_KEY environment variable [1]
    free_model = GeminiModel(model="gemini-2.5-flash")
    
    test_case = LLMTestCase(
        input=user_query,
        actual_output=actual_bot_response,
        retrieval_context=[f"We currently have 50 units of {product_name} in stock."]
    )
    
    # Pass the free model directly into the metric to avoid OpenAI default errors [1]
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=free_model)
    
    # This assertion will fail the entire Pytest run if the AI hallucinates
    assert_test(test_case, [relevancy_metric])