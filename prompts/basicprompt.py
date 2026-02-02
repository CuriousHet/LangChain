from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# template="Explain {topic} in simple termns."
template = """
You are a senior software engineer.

Task:
Explain {concept}

Constraints:
- Use bullet points
- Add one real-world analogy
- Keep under {word_limit} words
"""

prompt = PromptTemplate(
    input_variables=["concept", "word_limit"],
    template=template,
)

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)
formatted_prompt = prompt.format(concept="Neural networks", word_limit=100)

res = model.invoke(formatted_prompt)
print(res.content)