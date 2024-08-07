from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

class Questioner():

    def __init__(self, HUGGINGFACEHUB_API_TOKEN: str | None = None) -> None:
        # if HUGGINGFACEHUB_API_TOKEN is None, then it will look the enviroment variable HUGGINGFACEHUB_API_TOKEN

        self.llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.2')
        #meta-llama/Meta-Llama-3-8B-Instruct
        #mistralai/Mistral-7B-Instruct-v0.2

        self.parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant for suggesting appropriate questions to ask vendors during freight procurement. These questions should help evaluate the vendors' capabilities effectively.

            Generic Factors:
            1. Experience
            2. References
            3. Delivery Times
            4. Safety and Security
            5. Rates and Payment Terms
            6. Customs Clearance
            7. Tracking Services
            8. Capacity
            9. Handling Hazardous Materials
            10. Insurance
            11. Sustainability
            12. Revenue
            13. Technology
            14. Customer Service
            15. Compliance

            Modes and Delivery Types:
            - Ocean: FCL (Full Container Load), LCL (Less than Container Load)
            - Air: Cargo, Courier, Parcel
            - Road: FTL (Full Truck Load), PTL/LTL (Partial Truck Load or Less than Truck Load), Courier, Parcel

            For each mode and delivery type, suggest 5 to 10 unique questions tagged with appropriate factors. Ensure that no question is similar to those already provided in the template.

            Examples:
            Input: mode: 'Ocean', delivery_type: 'FCL', existing_questions: ['What is your experience with FCL shipments?']
            Output: 
            - Can you provide references from previous clients for FCL shipments? (References)
            - What are your standard delivery times for FCL and how do you handle delays? (Delivery Times)
            - How do you ensure the safety and security of FCL shipments? (Safety and Security)
            - What are your rates and payment terms for FCL shipments? (Rates and Payment Terms)
            - How do you handle customs clearance for FCL shipments? (Customs Clearance)
            - Do you offer tracking services for FCL shipments? (Tracking Services)
            - What is your capacity for handling high-volume FCL shipments? (Capacity)
            - How do you handle hazardous materials in FCL shipments? (Handling Hazardous Materials)
            - Can you provide information on your insurance coverage for FCL shipments? (Insurance)
            - What measures do you take to ensure sustainability in FCL shipments? (Sustainability)

            Do not include any explanations or additional text in your response.
            """),
            ("human", "mode: '{mode}', delivery_type: '{delivery_type}', existing_questions: {existing_questions}")])

        self.chain = self.prompt | self.llm | self.parser

    def suggest_questions(self, mode, delivery_type, existing_questions):
        existing_questions_str = str(existing_questions)
        input_data = {
            "mode": mode,
            "delivery_type": delivery_type,
            "existing_questions": existing_questions_str
        }
        response = self.chain.invoke(input_data)
        questions = [line.replace('- ', '').strip() for line in response.strip().split("\n") if '-' in line]
        return questions