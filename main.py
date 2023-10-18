# Standard library
import os

# Third-party libraries
from dotenv import load_dotenv
import openai

# langchain imports
from langchain.chains import LLMChain
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
            "E": "English",
        }
        self.text_to_translate = "What's up, Doc?"
        self.chat = ChatOpenAI(
            temperature=0,
            model="gpt-4"
            #model="gpt-3.5-turbo"
        )
        system_template = ("You are Skippy the Magnificent! A beer-can sized, quick-witted, highly sarcastic, "
                           "pain in the ass ancient AI from Craig Alanson's Expeditionary Force series whose intellect "
                           "is as big as your ego -- which exceeds the size of the sun. Your sarcastic demeanor and vast "
                           "knowledge make for quite the character. I want to share your response with others, so instead "
                           "of translating what I send your way, please translate your witty response from {input_language} "
                           "to {output_language}.")
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = "{text}"
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages([self.system_message_prompt, self.human_message_prompt])
        self.chain = LLMChain(llm=self.chat, prompt=self.chat_prompt)

    def translate(self, output_lang_key):
        output_lang = self.output_langs[output_lang_key]
        skippy_response = self.chain.run(input_language=self.input_lang,
                                         output_language="English",
                                         text=self.text_to_translate)
        translated_response = self.chain.run(input_language=self.input_lang,
                                             output_language=output_lang,
                                             text=skippy_response)
        return translated_response

    def translate_all(self):
        translations = {}
        for key, lang in self.output_langs.items():
            translations[lang] = self.translate(key)
        return translations

    def translate_each(self):
        translations = {}
        for key, lang in self.output_langs.items():
            user_input = input(f"Do you want to translate to {lang}? (yes/no): ")
            if user_input.lower() in ['yes', 'y']:
                translations[lang] = self.translate(key)
                print(f'Translation ({lang}): {translations[lang]}')
        return translations

class LanguageSelector:
    def __init__(self):
        self.languages = self._fetch_languages()

    def _fetch_languages(self):
        llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
        template = "List a variety of programming languages including common, uncommon, old, and esoteric ones."
        chat_prompt = ChatPromptTemplate.from_messages([("system", template)])
        # Get the list of messages from the ChatPromptTemplate object
        messages = chat_prompt.messages
        response = llm.invoke({"messages": messages})
        languages = response['choices'][0]['message']['content'].split(", ")
        return languages

    def display_languages(self):
        for idx, lang in enumerate(self.languages, 1):
            print(f"{idx}. {lang}")

    def get_language_by_index(self, index):
        return self.languages[index - 1]

class LoopCodeGenerator:
    def __init__(self):
        self.llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.languages = ["Python", "Java", "C#", "JavaScript", "Brainfuck"]  # Hardcoded list of languages

    def generate_loop_code(self, text, repetitions, language):
        template = (f"You are a helpful assistant who generates loop code. "
                    f"Generate a loop in {language} that prints the text '{text}' {repetitions} times.")
        messages = [{"role": "system", "type": "system", "content": template}]
        response = self.llm.invoke(messages)
        return response['choices'][0]['message']['content']

    def prompt_user(self):
        text = "I will not use GPT to do my homework"
        repetitions = 500
        print("Available languages:")
        for idx, lang in enumerate(self.languages, 1):
            print(f"{idx}. {lang}")
        language_index = int(input("Choose a language from the list above (enter the number): "))
        language = self.languages[language_index - 1]
        loop_code = self.generate_loop_code(text, repetitions, language)
        print("\nGenerated Loop Code:\n")
        print(loop_code)

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
    return translator.translate_each()


def main():
    name = 'Doctor Ew'
    print(f'Hello there, {name}')

    loop_generator = LoopCodeGenerator()
    loop_generator.prompt_user()

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
