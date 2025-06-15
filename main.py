from typing_extensions import TypedDict
from openai import OpenAI
from langgraph.graph import  StateGraph,START,END
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

class State(TypedDict):
    query : str
    llm_result : str


def chat_bot(state : State):
    query = state['query']
    llm_response = client.chat.completions.create(
        model = "chatgpt-4o-latest",
        messages={
            'role':"user","content":query
        }
    )
    
    state['llm_result']=llm_response
    
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("chat_bot",chat_bot)

graph_builder.add_edge(START,"chat_bot")
graph_builder.add_edge(chat_bot",END)

graph = graph_builder.compile()

def main():
    query = input(">")
    
    _state = {
        "query":query,
        "llm_result":None
    }
    
    graph_result = graph.index(_state)
    
    print("result:",graph_result)
    

main()
