from typing_extensions import TypedDict
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()


class ResumeCheckResponse(BaseModel):
    is_resume: bool

class ResumeScoreResponse(BaseModel):
    score: int
    suggestions: str

class ResumeState(TypedDict):
    user_query: str
    is_resume: bool | None
    critique: str | None
    score: int | None

client = OpenAI()

def check_if_resume(state: ResumeState):
    print("🧾 check_if_resume")
    SYSTEM_PROMPT = """
    You're an AI assistant who determines whether a given input text is a resume.
    Return: {"is_resume": true/false}
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        response_format=ResumeCheckResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["user_query"]},
        ]
    )
    state["is_resume"] = response.choices[0].message.parsed.is_resume
    return state


def critique_resume(state: ResumeState):
    print("📊 critique_resume")
    SYSTEM_PROMPT = """
    You are a professional resume reviewer. Provide an honest critique of this resume.
    Evaluate formatting, grammar, clarity, conciseness, skills listed, and structure.
    Return score out of 100 and 2-3 actionable improvement points.
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4.1",
        response_format=ResumeScoreResponse,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["user_query"]},
        ]
    )
    parsed = response.choices[0].message.parsed
    state["score"] = parsed.score
    state["critique"] = parsed.suggestions
    return state


def general_handler(state: ResumeState):
    print("🗨️ general_handler")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": state["user_query"]}]
    )
    state["critique"] = response.choices[0].message.content
    return state


builder = StateGraph(ResumeState)

builder.add_node("check_if_resume", check_if_resume)
builder.add_node("critique_resume", critique_resume)
builder.add_node("general_handler", general_handler)

def router(state: ResumeState) -> Literal["critique_resume", "general_handler"]:
    return "critique_resume" if state["is_resume"] else "general_handler"

builder.add_edge(START, "check_if_resume")
builder.add_conditional_edges("check_if_resume", router)

builder.add_edge("critique_resume", END)
builder.add_edge("general_handler", END)

resume_graph = builder.compile()

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def main():
    choice = input("Enter 1 to paste resume, 2 to upload a PDF file: ")

    if choice == "1":
        user_input = input("📄 Paste your resume or question: ")
    elif choice == "2":
        path = input("📂 Enter path to your resume (PDF): ")
        user_input = extract_text_from_pdf(path)
        print("✅ Extracted resume text.")
    else:
        print("❌ Invalid option.")
        return

    initial_state: ResumeState = {
        "user_query": user_input,
        "is_resume": None,
        "critique": None,
        "score": None
    }

    for event in resume_graph.stream(initial_state):
        print("Event", event)

    print("\n📊 Final Output:")
    if initial_state["score"] is not None:
        print(f"Score: {initial_state['score']}")
    print("Critique:", initial_state["critique"])


main()