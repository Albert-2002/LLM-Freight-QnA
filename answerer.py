from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Answerer():

    def __init__(self, HUGGINGFACEHUB_API_TOKEN: str | None = None) -> None:
        #HUGGINGFACEHUB_API_TOKEN
        #OPENAI_API_KEY

        self.llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.3, top_p=0.9)
        # self.llm = OpenAI(model="gpt-3.5-turbo-instruct")

        self.prompt_proper = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant for providing proper answers to vendor questions in the context of freight procurement. Provide a direct, detailed, and accurate answer to the given question based on industry best practices and common knowledge. Ensure the answers are quantitative and showcase expertise. Start every answer with "Proper Answer: ...". Do not include any explanations, additional text, or follow-up questions. Only respond to the question asked.

            Examples:
            Question: What is your experience with FCL shipments?
            Proper Answer: "We have over 15 years of experience handling Full Container Load (FCL) shipments, completing more than 1,000 successful deliveries annually. Our team is certified in international logistics management, and we leverage a network of over 50 reliable carriers. Our advanced tracking systems provide real-time updates and have reduced delays by 20% over the past five years."

            Question: How do you ensure the safety and security of FCL shipments?
            Proper Answer: "We implement stringent safety and security measures for FCL shipments, including ISO 28000 certification for supply chain security management. Our protocols feature secure packaging, tamper-evident seals, 24/7 real-time GPS tracking, and compliance with international shipping regulations. Regular audits and quarterly training sessions ensure our staff are proficient in the latest security practices. Our incident rate is below 0.5%, demonstrating our commitment to safety."
            """),
            ("human", "{input}")
        ])

        self.prompt_poor = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant for providing poor answers to vendor questions in the context of freight procurement. Provide a direct answer to the given question that shows some effort but lacks detail, specificity, and confidence. Start every answer with "Poor Answer: ...". Do not include any explanations, additional text, or follow-up questions. Only respond to the question asked.

            Examples:
            Question: What is your experience with FCL shipments?
            Poor Answer: "We have handled FCL shipments for a few years now. Our team is generally experienced, and we have a number of successful shipments. We work with several carriers and use tracking systems."

            Question: How do you ensure the safety and security of FCL shipments?
            Poor Answer: "We try to ensure the safety and security of our shipments by following standard procedures. We use basic packaging materials and have some tracking in place. Our staff is trained, and we follow most of the shipping regulations."
            """),
            ("human", "{input}")
        ])

        self.output_parser = StrOutputParser()

        self.chain_proper = self.prompt_proper | self.llm | self.output_parser
        self.chain_poor = self.prompt_poor | self.llm | self.output_parser

    def provide_proper_answer(self, question):
        response = self.chain_proper.invoke({"input": question})
        return response.strip().replace("Human:", "").strip()

    def provide_poor_answer(self, question):
        response = self.chain_poor.invoke({"input": question})
        return response.strip().replace("Human:", "").strip()