# Standard library
import os

# Third-party libraries
from dotenv import load_dotenv
import openai

# langchain imports
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, BaseOutputParser, Document
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.document_transformers import DoctranTextTranslator

load_dotenv()


class TextTranslator:
    def __init__(self):
        self.llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.chat_model = ChatOpenAI()
        self.input_lang = "English"
        self.output_langs = {
            "S": "Spanish",
            "H": "Hebrew",
            "Y": "Yiddish",
            "K": "Klingon",
        }
        self.text_to_translate = "You are a helpful assistant that translates {input_language} to {output_language}"
        self.chat = ChatOpenAI(
            temperature=0,
            model="gpt-4"
            # model="gpt-3.5-turbo-0613"
        )
        system_template = "You are a helpful assistant that translates {input_language} to {output_language}."
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = "{text}"
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages([self.system_message_prompt, self.human_message_prompt])
        self.chain = LLMChain(llm=self.chat, prompt=self.chat_prompt)

    def translate(self, output_lang_key):
        output_lang = self.output_langs[output_lang_key]
        return self.chain.run(input_language=self.input_lang,
                              output_language=output_lang,
                              text=self.text_to_translate)

    def translate_all(self):
        translations = {}
        for key, lang in self.output_langs.items():
            translations[lang] = self.translate(key)
        return translations


#

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
    translator = TextTranslator()
    return translator.translate_all()


def main():
    name = 'Doctor Ew'
    print(f'Hello there, {name}')

    text = "What would be a good company name for a company that makes colorful socks?"
    messages = [HumanMessage(content=text)]

    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    chat_model = ChatOpenAI()

    prediction_llm = llm.predict_messages(messages)
    prediction_cm = chat_model.predict_messages(messages)

    print(f'Prediction: {prediction_llm} vs {prediction_cm}')

    translations = translate_text()
    print(f'Original (English): {text}')
    for lang, translation in translations.items():
        print(f'Translation ({lang}): {translation}')

    print(f'Comma-separated list: {generate_comma_separated_list()}')


if __name__ == '__main__':
    main()
