from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Answerer:

    def __init__(self, HUGGINGFACEHUB_API_TOKEN: str | None = None) -> None:

        self.llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")

        self.prompt_proper = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant for providing proper answers to vendor questions in the context of freight procurement. Provide detailed, relevant, and accurate answers based on industry best practices and common knowledge.

            Examples:
            Question: What is your experience with FCL shipments?
            Proper Answer: We have over 15 years of experience handling Full Container Load (FCL) shipments. Our team is well-versed in managing large-scale logistics operations, ensuring timely delivery and maintaining the integrity of the goods. We work with a network of reliable carriers and use advanced tracking systems to monitor shipments from origin to destination.

            Question: How do you ensure the safety and security of FCL shipments?
            Proper Answer: We implement stringent safety and security measures for FCL shipments. Our protocols include secure packaging, tamper-evident seals, real-time GPS tracking, and compliance with international shipping regulations. Additionally, we conduct regular audits and training sessions to keep our staff updated on the latest security practices.

            Do not include any explanations or additional text in your response.
            """),
            ("human", "{input}")
        ])

        self.prompt_poor = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant for providing poor answers to vendor questions in the context of freight procurement. Provide vague, irrelevant, or inaccurate answers that do not inspire confidence in the vendor's capabilities.

            Examples:
            Question: What is your experience with FCL shipments?
            Poor Answer: We have done some FCL shipments in the past. Not too many problems usually. It depends on the situation, but we try to do our best.

            Question: How do you ensure the safety and security of FCL shipments?
            Poor Answer: We usually make sure things are safe. Our staff is trained, and we try to follow the rules most of the time. Sometimes things go wrong, but we handle it.

            Do not include any explanations or additional text in your response.
            """),
            ("human", "{input}")
        ])

        self.output_parser = StrOutputParser()

        self.chain_proper = self.prompt_proper | self.llm | self.output_parser
        self.chain_poor = self.prompt_poor | self.llm | self.output_parser

    def provide_proper_answer(self, question):
        response = self.chain_proper.invoke({"input": question})
        return response.strip()

    def provide_poor_answer(self, question):
        response = self.chain_poor.invoke({"input": question})
        return response.strip()