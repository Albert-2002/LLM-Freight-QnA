from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

class Questioner():

    def __init__(self, HUGGINGFACEHUB_API_TOKEN: str | None = None) -> None:
        # if HUGGINGFACEHUB_API_TOKEN is None, then it will look the enviroment variable HUGGINGFACEHUB_API_TOKEN
        # OPENAI_API_KEY

        self.llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.2')
        # self.llm = OpenAI(model="gpt-3.5-turbo-instruct")
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

            Input: mode: 'Ocean', delivery_type: 'FCL', existing_questions: ['Can you provide references from previous clients for FCL shipments?', 'What are your standard delivery times for FCL and how do you handle delays?', 'How do you ensure the safety and security of FCL shipments?', 'What are your rates and payment terms for FCL shipments?', 'How do you handle customs clearance for FCL shipments?', 'Do you offer tracking services for FCL shipments?', 'What is your capacity for handling high-volume FCL shipments?', 'How do you handle hazardous materials in FCL shipments?', 'Can you provide information on your insurance coverage for FCL shipments?', 'What measures do you take to ensure sustainability in FCL shipments?']
            Output:
            - How many years of experience do you have specifically in handling FCL shipments? (Experience)
            - Can you provide case studies or examples of successfully handled FCL shipments? (References)
            - What is the average transit time for FCL shipments to major global destinations? (Delivery Times)
            - What protocols do you have in place to prevent theft or damage during FCL shipping? (Safety and Security)
            - Are there any additional fees or surcharges for FCL shipments during peak seasons? (Rates and Payment Terms)
            - How do you manage and expedite customs clearance issues for FCL shipments? (Customs Clearance)
            - Can customers track FCL shipments in real-time? (Tracking Services)
            - How do you scale your services to accommodate large FCL shipment volumes? (Capacity)
            - What safety measures are taken for hazardous materials in FCL shipping? (Handling Hazardous Materials)
            - What types of insurance coverage are available for FCL shipments? (Insurance)

            Input: mode: 'Ocean', delivery_type: 'FCL', existing_questions: ['How many years of experience do you have specifically in handling FCL shipments?', 'Can you provide case studies or examples of successfully handled FCL shipments?', 'What is the average transit time for FCL shipments to major global destinations?', 'What protocols do you have in place to prevent theft or damage during FCL shipping?', 'Are there any additional fees or surcharges for FCL shipments during peak seasons?', 'How do you manage and expedite customs clearance issues for FCL shipments?', 'Can customers track FCL shipments in real-time?', 'How do you scale your services to accommodate large FCL shipment volumes?', 'What safety measures are taken for hazardous materials in FCL shipping?', 'What types of insurance coverage are available for FCL shipments?']
            Output:
            - What technology do you use to monitor and optimize FCL shipments? (Technology)
            - How do you ensure compliance with international shipping regulations for FCL shipments? (Compliance)
            - What is your procedure for handling customer complaints regarding FCL shipments? (Customer Service)
            - How do you manage the risk of container shortages for FCL shipments? (Capacity)
            - What are your policies regarding the sustainability and environmental impact of FCL shipments? (Sustainability)
            - How do you handle emergency situations or disruptions in FCL shipments? (Safety and Security)
            - Can you provide a detailed breakdown of your FCL shipping costs? (Rates and Payment Terms)
            - How do you integrate new technology to improve FCL shipment efficiency? (Technology)
            - What is your process for regular maintenance and inspection of containers used in FCL shipments? (Safety and Security)
            - How do you ensure effective communication with clients throughout the FCL shipping process? (Customer Service)

            Do not include any explanations or additional text in your response.
            """),
            ("human", "mode: '{mode}', delivery_type: '{delivery_type}', existing_questions: {existing_questions}")])

        self.chain = self.prompt | self.llm | self.parser

    def suggest_questions(self, mode, delivery_type, existing_questions):
        default_questions = self.get_default_questions(mode, delivery_type)
        combined_questions = default_questions + existing_questions

        existing_questions_str = str(combined_questions)
        input_data = {
            "mode": mode,
            "delivery_type": delivery_type,
            "existing_questions": existing_questions_str
        }
        response = self.chain.invoke(input_data)
        lines = response.strip().split("\n")

        suggested_questions = [
            line.replace('- ', '').strip()
            for line in lines
            if '-' in line and not line.startswith(('AI:', 'Suggested:', 'Examples:', 'Input:', 'Output:'))]

        return {
            "existing_questions": combined_questions,
            "suggested_questions": suggested_questions
        }

    def get_default_questions(self, mode, delivery_type):
        defaults = {
            ("Ocean", "FCL"): [
                "What is your experience with FCL shipments?",
                "Can you provide references from previous clients for FCL shipments?",
                "What are your standard delivery times for FCL and how do you handle delays?"
            ],
            ("Air", "Cargo"): [
                "What is your experience with air cargo services?",
                "Can you provide references from previous clients for air cargo shipments?",
                "What are your standard delivery times for air cargo and how do you handle delays?"
            ],
            ("Road", "FTL"): [
                "What is your experience with FTL shipments?",
                "Can you provide references from previous clients for FTL shipments?",
                "What are your standard delivery times for FTL and how do you handle delays?",
                "How do you ensure the safety of the movement of all materials?",
                "What types of goods do you specialize in transporting?",
                "What qualifications and certifications do your drivers have?",
                "What types of insurance coverage do you carry for cargo and liability?",
                "What are your payment terms, and do you offer any discounts for volume or long-term contracts?"
            ],
            ("Air", "Courier"): [
                "Can you provide references from previous clients for Air Courier movements?",
                "What are your standard delivery times for Courier and how do you handle delays?",
                "Do you offer a tracking service for air couriers?",
                "What key performance indicators (KPIs) do you use to measure your service quality?",
                "Do you provide a dedicated account manager or point of contact?"
            ]
        }
        return defaults.get((mode, delivery_type), [])