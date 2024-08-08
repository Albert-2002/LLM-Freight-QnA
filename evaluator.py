import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

class Evaluator:
    def __init__(self, HUGGINGFACEHUB_API_TOKEN: str | None = None) -> None:
        # Use HuggingFace model or OpenAI model for the LLM
        self.llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.2')
        self.parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant for evaluating answers to questions about air courier services. 
            Evaluate the answer on a scale of 1 to 5 and provide a brief reasoning for the score.

            Example:
            Question: How will I be able to track my air courier package?
            Answer: We provide tracking information for air courier packages. You can check the status on our website or contact our customer service team for updates.
            Evaluation: 
            Score: 3
            Reasoning: The answer provides basic information but lacks detail on how the tracking system works and its reliability.

            Question: What specific measures do you implement to ensure on-time delivery for FCL shipments?
            Answer: We have a three-pronged strategy:
            1. Accurate Planning and Scheduling: Develop precise shipping schedules that account for transit times, port congestion, and potential delays.
            2. Real-Time Tracking and Visibility: Implement GPS and IoT-based tracking systems to monitor the location and status of shipments in real-time.
            3. Contingency Planning: Develop contingency plans for common issues such as port strikes, customs delays, or carrier capacity constraints.
            Evaluation: 
            Score: 5
            Reasoning: The answer provides a comprehensive and structured approach, addressing key areas essential for ensuring on-time delivery.

            Do not include any additional text in your response other than the score and reasoning.
            """),
            ("human", "Question: '{question}'\nAnswer: '{answer}'")])

        self.chain = self.prompt | self.llm | self.parser

    def _parse_text(self, text):
        q_and_a = []
        lines = text.split('\n')
        question, answer = None, None

        for line in lines:
            if line.startswith("Question: "):
                if question and answer:
                    q_and_a.append((question, answer))
                question = line[len("Question: "):]
                answer = []
            elif line.startswith("Poor Answer: ") or line.startswith("Answer: "):
                answer = [line[len("Poor Answer: "):] if line.startswith("Poor Answer: ") else line[len("Answer: "):]]
            elif line.startswith("Proper Answer: "):
                answer = [line[len("Proper Answer: "):]]
            elif answer is not None:
                answer.append(line.strip())

        if question and answer:
            q_and_a.append((question, "\n".join(answer)))

        return q_and_a

    def evaluate(self, text):
        evaluations = []
        q_and_a = self._parse_text(text)
        for question, answer in q_and_a:
            score, reasoning = self._evaluate_qa(question, answer)
            evaluations.append({
                "question": question,
                "answer": answer,
                "score": score,
                "reasoning": reasoning
            })
        return evaluations

    def _evaluate_qa(self, question, answer):
        input_data = {
            "question": question,
            "answer": answer
        }
        response = self.chain.invoke(input_data)
        lines = response.strip().split("\n")
        score_line = next((line for line in lines if "Score:" in line), None)
        reasoning_line = next((line for line in lines if "Reasoning:" in line), None)

        if score_line and reasoning_line:
            score = int(score_line.split("Score: ")[1].strip())
            reasoning = reasoning_line.split("Reasoning: ")[1].strip()
            return score, reasoning
        else:
            return 1, "Could not determine a proper score and reasoning from the LLM response."

# Usage example
if __name__ == "__main__":
    with open('custom_answers.txt', 'r') as file:
        text = file.read()

    evaluator = Evaluator(HF_TOKEN)
    evaluations = evaluator.evaluate(text)

    total_score = 0
    count = 0

    for evaluation in evaluations:
        print(f"Question: {evaluation['question']}")
        print(f"Answer: {evaluation['answer']}")
        print(f"Score: {evaluation['score']}")
        print(f"Reasoning: {evaluation['reasoning']}")
        print("------")
        total_score += evaluation['score']
        count += 1

    average_score = total_score / count if count > 0 else 0
    print(f"Average Score: {average_score:.2f} out of 5")