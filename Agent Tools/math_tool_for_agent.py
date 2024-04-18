from langchain.agents import AgentType, Tool, initialize_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
# For Web Search Tool (Note: Bing Search API key is required to be set in env as BING_SUBSCRIPTION_KEY)
from langchain.utilities.bing_search import BingSearchAPIWrapper
# For Math Calculator Tool
from langchain.chains import LLMMathChain

import os
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

llm = ChatOpenAI(
    #base_url="http://localhost:1234/v1/",
    temperature=0
)

tools = [   
    Tool(
        name="Web Search",
        func=BingSearchAPIWrapper().run,
        description="useful for when you need to answer specific questions from information on the web",
    ),
    Tool(name="Math Calculator",
         func=LLMMathChain.from_llm(llm),
         description="useful for when you need to answer math questions"
    )
]

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

pydict = agent.invoke({"input": "Hi, I am Bob"})
print(pydict["output"])
pydict = agent.invoke({"input": "What's my name?"})
print(pydict["output"])
# Web searching
pydict = agent.invoke({"input": "Who is the CEO of LinkedIn in 2023?"})
print(pydict["output"])
# Math calculating
pydict = agent.invoke({"input": "What is the square root of 840889"})
print(pydict["output"])
