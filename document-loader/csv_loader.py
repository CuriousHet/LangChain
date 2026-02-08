from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

loader = CSVLoader(file_path='demo.csv')
docs = loader.load()
# print(len(docs))
# print(docs[0])

csv_text = "\n\n".join(doc.page_content for doc in docs)
# print(csv_text)

#since data is small, directly passing to model, incase if data is large do embedding + retreival

prompt = PromptTemplate(
    template='which of the employees are from {location} in data {csv_text}',
    input_variables=['location', 'csv_text']
)
parser = StrOutputParser()
chain = prompt | model | parser

print(chain.invoke({'location':'USA', 'csv_text': csv_text}))