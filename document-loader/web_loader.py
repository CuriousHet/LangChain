from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://docs.langchain.com/oss/python/langchain/agents'
loader = WebBaseLoader(url)

docs = loader.load()
chain = prompt | model | parser

print(chain.invoke({'question':'what does this page describes?', 'text':docs[0].page_content}))