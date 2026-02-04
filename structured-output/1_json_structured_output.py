# Prompt-only JSON is NOT reliable in production

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = """
    You are an AI analyst.

    You are an AI analyst.
    Analyze the following text.

    Return ONLY valid JSON using this schema:
    {{
    "product": string,
    "price": list[string],
    }}

    Text:
    {text}
"""
prompt = PromptTemplate(
    input_variables=["text"],
    template = template
)
messages = prompt.format(
    text="The iPhone 15 costs 799 dollars and is currently in stock."
)

res = model.invoke(messages)
print(res.content)