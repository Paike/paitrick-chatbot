from langchain.prompts import PromptTemplate

import os
import sys

file_path = 'prompt_template.txt'

# Check if the file exists
if os.path.isfile(file_path):
    # Open and read the file into a string
    with open(file_path, 'r') as file:
        cv_assistant_template = file.read()

else:
    # print(f"The prompt template file '{file_path}' does not exist.")
    # sys.exit(1)  # Exit with a non-zero status code to indicate an error
    cv_assistant_template = """
    You are a helpful assistant.

    This is the context: {context}

    This is the chat history: {history}
    
    Question: {question}
    Answer: """


cv_assistant_prompt_template = PromptTemplate(
    template=cv_assistant_template,
    input_variables=[
        'context',
        'history'
        'question',
    ]
)
