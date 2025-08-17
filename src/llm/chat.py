from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


def build_conversation_chain(llm, k: int = 3):
    memory = ConversationBufferWindowMemory(k=k, return_messages=True)
    return ConversationChain(memory=memory, prompt=None, llm=llm, verbose=False), memory
