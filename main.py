import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser

load_dotenv()

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
chat_model = ChatOpenAI()


class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")


def generate_comma_separated_list():
    template = """You are a helpful assistant who generates comma separated lists.
    A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
    ONLY return a comma separated list, and nothing more."""
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()
    return chain.invoke({"text": "colors"})


def translate_text():
    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    return chat_prompt.format_messages(input_language="English", output_language="French",
                                       text="I love programming.")


def main():
    name = 'PyCharm'
    print(f'Hi, {name}')

    text = "What would be a good company name for a company that makes colorful socks?"
    messages = [HumanMessage(content=text)]

    prediction_llm = llm.predict_messages(messages)
    prediction_cm = chat_model.predict_messages(messages)

    print(f'Prediction: {prediction_llm} vs {prediction_cm}')
    print(f'translation: {translate_text()}')

    print(f'Comma-separated list: {generate_comma_separated_list()}')


if __name__ == '__main__':
    main()
