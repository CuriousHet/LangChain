from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_prompt = ChatPromptTemplate([
    ("system", "You are a tutor"),
    ("human", "Explain {topic} with examples")
])

messages = chat_prompt.format_messages(topic="few shot prompting")
res = model.invoke(messages)

print(res.content)