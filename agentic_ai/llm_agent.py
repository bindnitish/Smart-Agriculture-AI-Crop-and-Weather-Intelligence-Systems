from openai import OpenAI
import json


client = OpenAI()


def make_llm_decision(crop, weather, crop_confidence=None, weather_confidence=None):

    prompt = f"""
You are an agricultural decision-making AI agent.

Your task is to make a context-aware agricultural decision
using ONLY the information provided below.

Crop:
{crop}

Crop confidence:
{crop_confidence}

Weather:
{weather}

Weather confidence:
{weather_confidence}

Important rules:
1. Do not diagnose diseases.
2. Do not invent information that is not provided.
3. Consider both crop and weather together.
4. Give a practical agricultural recommendation.
5. Select an appropriate IoT action.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text