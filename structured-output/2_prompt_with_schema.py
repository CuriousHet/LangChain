# Prompt constraints help, but still not safe enough

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a strict JSON generator.

Rules:
- Output MUST be valid JSON
- Follow schema exactly
- Do not add extra fields
- Do not explain
"""),
    ("human", """
Schema:
{{
  "name": string,
  "age": integer,
  "skills": list[string]
}}

Input:
{input}
""")
])

messages = prompt.format_messages(
    input="John is 25 years old and knows Python and Go"
)

res = model.invoke(messages)
print(res.content)