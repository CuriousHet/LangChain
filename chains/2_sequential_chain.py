from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'generate a report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='summaries the {report}',
    input_variables=['report']
)

chain = prompt1 | model | parser | prompt2 | model | parser
res = chain.invoke({'topic':'indian cricket team'})
print(res)

chain.get_graph().print_ascii()


