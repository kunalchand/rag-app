from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


def conversation_prompt():
    system_msg_template = SystemMessagePromptTemplate.from_template(
        'Answer truthfully using the provided context. If not found, say "I don\'t know" and suggest uploading relevant PDFs.'
    )
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    return ChatPromptTemplate.from_messages(
        [
            system_msg_template,
            MessagesPlaceholder(variable_name="history"),
            human_msg_template,
        ]
    )
