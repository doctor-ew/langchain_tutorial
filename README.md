# LangChain Python Initial Tests

This repository contains a Python script that demonstrates the capabilities of the `langchain` package, specifically its integration with OpenAI. The script showcases various functionalities such as generating comma-separated lists, translating text, and predicting messages.

The script combines concepts and techniques from two tutorials:
1. [The official LangChain Quickstart documentation](https://python.langchain.com/docs/get_started/quickstart#environment-setup).
2. [Build a GPT-powered translator with LangChain](https://levelup.gitconnected.com/build-a-gpt-powered-translator-with-langchain-3e6915914daf)

## Getting Started

### Prerequisites

1. Python 3.x
2. A virtual environment tool like `venv` or `virtualenv`.
3. An OpenAI API key.

### Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/your_repository_name.git
   cd your_repository_name
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Required Packages**:
   ```bash
   pip install python_dotenv langchain openai doctran virtualfish
   ```
   - note that Doctor Ew like any good mad scientist doesn't run the "normal" shells, so `virtualfish` or Pycharm will help set up the proper environment

4. **Environment Configuration**:
   - Create a `.env` file in the root directory of the project.
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

### Running the Script

With the virtual environment activated and the `.env` file set up, you can run the script:

```bash
python main.py
```

This will execute the main function, which demonstrates various functionalities of the `langchain` package.

### Results

After following the tutorial and executing the script, you should see results similar to the ones shown below:

![screenshot](https://www.doctorew.com/shuttlebay/Screen-Shot-2023-10-16-17-17-52.07.png)

## Further Reading

For a more detailed guide on setting up and using the `langchain` package, please refer to the [official LangChain Quickstart documentation](https://python.langchain.com/docs/get_started/quickstart#environment-setup).

---

* Footnote: This README was crafted with the assistance of my GPT Assistant, A.C.E. (Artificial Chat Engineer).

