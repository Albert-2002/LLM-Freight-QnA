from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

class Questioner():

    def __init__(self, HUGGINGFACEHUB_API_TOKEN: str | None = None) -> None:
        # if HUGGINGFACEHUB_API_TOKEN is None, then it will look the enviroment variable HUGGINGFACEHUB_API_TOKEN

        self.llm = HuggingFaceEndpoint(repo_id='meta-llama/Meta-Llama-3-8B-Instruct')
        self.parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant for suggesting appropriate questions to ask vendors during freight procurement. These questions should help evaluate the vendors' capabilities effectively.

            Examples of questions to ask vendors:
            1. What is your experience with similar freight shipments?
            2. Can you provide references from previous clients?
            3. What are your standard delivery times and how do you handle delays?
            4. What measures do you take to ensure the safety and security of the freight?
            5. What are your rates and payment terms?
            6. How do you handle customs clearance and documentation?
            7. Do you offer tracking services for shipments? If so, how does it work?
            8. What is your capacity for handling high-volume shipments?
            9. How do you handle hazardous materials?
            10. Can you provide information on your insurance coverage?

            Provide a set of appropriate questions based on the given context.
            """),
            ("human", "{input}")])

        self.chain = self.prompt | self.llm | self.parser

    def suggest_questions(self, context):
        response = self.chain.invoke({"input": context})
        questions = response.strip().split("\n")
        return questions