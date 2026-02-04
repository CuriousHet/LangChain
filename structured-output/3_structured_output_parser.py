from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

schemas = [
    ResponseSchema(name="language", description="Programming language"),
    ResponseSchema(name="difficulty", description="beginner/intermediate/advanced"),
    ResponseSchema(name="use_case", description="Where it is used")
]

parser = StructuredOutputParser.from_response_schemas(schemas)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert programmer."),
    ("human", """
        Explain {topic}

        {format_instructions}
    """)
])

messages = prompt.format_messages(
    topic="Golang",
    format_instructions=parser.get_format_instructions()
)

res = model.invoke(messages)
parsed = parser.parse(res.content)

print(parsed)
