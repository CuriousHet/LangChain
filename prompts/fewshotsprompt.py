from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

examples = [
    {"input": "awesome product", "sentiment":"positive"},
    {"input": "this is terrible", "sentiment":"negative"},
]

prompt = PromptTemplate(
    input_variables=["input", "sentiment"],
    template="Text: {input}\nSentiment: {sentiment}",
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt,
    prefix="Classify the sentiment:",
    suffix="Text: {text}\nSentiment:",
    input_variables=["text"]
)


final_prompt = few_shot_prompt.format(text="The service was bad")
print(model.invoke(final_prompt).content)