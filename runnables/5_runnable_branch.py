from langchain.schema.runnable import (
    RunnableLambda,
    RunnableBranch
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

def classify_intent(input: dict) -> str:
    query = input["query"].lower()

    if any(word in query for word in ["what", "who", "when", "where"]):
        return "fact"
    if any(word in query for word in ["explain", "why", "how"]):
        return "explanation"
    return "other"

fact_prompt = PromptTemplate(
    template="Give a concise factual answer to: {query}",
    input_variables=["query"]
)

explain_prompt = PromptTemplate(
    template="Explain the following in a clear, beginner-friendly way:\n{query}",
    input_variables=["query"]
)


intent_classifier = RunnableLambda(classify_intent)
fact_chain = fact_prompt | model | parser
explain_chain = explain_prompt | model | parser
fallback_chain = RunnableLambda(
    lambda x: "I'm not sure how to handle this request."
)

branch = RunnableBranch(
    (lambda x: x["intent"] == "fact", fact_chain),
    (lambda x: x["intent"] == "explanation", explain_chain),
    fallback_chain
)

chain = (
    RunnableLambda(
        lambda x: {
            **x,
            "intent": classify_intent(x)
        }
    )
    | branch
)

print(chain.invoke({"query": "What is quantum computing?"}))
print(chain.invoke({"query": "Explain how photosynthesis works"}))
print(chain.invoke({"query": "Write me a poem"}))

