import os
import operator
import base64
from uuid import uuid4
from typing import TypedDict

import streamlit as st
from openai import OpenAI
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import Annotated

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = init_chat_model("openai:gpt-4o-mini")


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class State(TypedDict):
    word_promt: str
    thumbnail_prompts: Annotated[list[str], operator.add]
    thumbnail_sketches: Annotated[list[bytes], operator.add]
    final_thumbnail: bytes
    final_summary: str
    user_feedback: str
    chosen_prompt: str


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def mega_summary(state: State):
    prompt = f"""
        When given the meaning of an English word, generate a prompt for creating a YouTube Shorts thumbnail.

        English word explanation:
        {state["word_promt"]}
    """
    response = llm.invoke(prompt)
    return {"final_summary": response.content}


def dispatch_artists(state: State):
    return [
        Send("generate_thumbnails", {"id": i, "summary": state["final_summary"]})
        for i in [1, 2, 3]
    ]


def generate_thumbnails(args):
    summary = args["summary"]

    prompt = f"""
    You are creating a YouTube Shorts thumbnail for an educational English vocabulary channel.
    The content is safe, family-friendly, and suitable for all ages.

    Your goal:
    Make the English word AND its etymology instantly understandable within 3 seconds.

    REQUIREMENTS:
    - The word must be LARGE, bold, and central (primary focus)
    - The etymology/root must be visually highlighted and connected to the meaning
    - The design must be SIMPLE, high contrast, and instantly readable on mobile
    - Use only positive, educational, non-violent imagery

    Extract from the summary:
    - Target word
    - Root / origin meaning
    - Core visual metaphor

    DESIGN STRUCTURE:
    1. Main Word (VERY LARGE, center)
    2. Root / Origin (smaller but highlighted, e.g., "pro = forward")
    3. Visual Metaphor (clearly shows the meaning of the root)

    VISUAL STYLE:
    - Bright, high contrast colors (yellow, blue, green, white)
    - Clean, well-lit scene
    - Minimal background, no clutter
    - Focus on ONE clear, family-friendly scene

    TEXT OVERLAY:
    - Max 3-5 words total
    - Example format: PROTECT / pro = forward

    IMPORTANT: Use only safe, educational, non-violent imagery appropriate for a language learning channel.

    Summary:
    {summary}
    """

    response = llm.invoke(prompt)
    thumbnail_prompt = f"Educational English vocabulary YouTube thumbnail, safe for all ages, family-friendly. {response.content[:3800]}"

    client = OpenAI()
    result = client.images.generate(
        model="gpt-image-1-mini",
        prompt=thumbnail_prompt,
        quality="low",
        size="1024x1536",
        output_format="jpeg",
    )
    image_bytes = base64.b64decode(result.data[0].b64_json)

    return {
        "thumbnail_prompts": [thumbnail_prompt],
        "thumbnail_sketches": [image_bytes],
    }


def human_feedback(state: State):
    answer = interrupt(
        {
            "chosen_thumbnail": "Which thumbnail do you like the most?",
            "feedback": "Provide any feedback or changes you'd like for the final thumbnail.",
        }
    )
    return {
        "user_feedback": answer["user_feedback"],
        "chosen_prompt": state["thumbnail_prompts"][answer["chosen_prompt"] - 1],
    }


def generate_hd_thumbnail(state: State):
    prompt = f"""
    You are a professional YouTube thumbnail designer. Take this original thumbnail prompt and create an enhanced version that incorporates the user's specific feedback.

    ORIGINAL PROMPT:
    {state["chosen_prompt"]}

    USER FEEDBACK TO INCORPORATE:
    {state["user_feedback"]}

    Create an enhanced prompt that:
        1. Maintains the core concept from the original prompt
        2. Specifically addresses and implements the user's feedback requests
        3. Adds professional YouTube thumbnail specifications:
            - High contrast and bold visual elements
            - Clear focal points that draw the eye
            - Professional lighting and composition
            - Optimal text placement and readability with generous padding from edges
            - Colors that pop and grab attention
            - Elements that work well at small thumbnail sizes
            - IMPORTANT: Always ensure adequate white space/padding between any text and the image borders
    """
    response = llm.invoke(prompt)
    final_thumbnail_prompt = f"Educational English vocabulary YouTube thumbnail, safe for all ages, family-friendly. {response.content[:3800]}"

    client = OpenAI()
    result = client.images.generate(
        model="gpt-image-1-mini",
        prompt=final_thumbnail_prompt,
        quality="high",
        size="1024x1536",
        output_format="jpeg",
    )
    image_bytes = base64.b64decode(result.data[0].b64_json)

    return {"final_thumbnail": image_bytes}


# ---------------------------------------------------------------------------
# Compile Graph (cached — runs once per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource
def compile_graph():
    memory = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node("mega_summary", mega_summary)
    builder.add_node("generate_thumbnails", generate_thumbnails)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("generate_hd_thumbnail", generate_hd_thumbnail)

    builder.add_edge(START, "mega_summary")
    builder.add_conditional_edges("mega_summary", dispatch_artists, ["generate_thumbnails"])
    builder.add_edge("generate_thumbnails", "human_feedback")
    builder.add_edge("human_feedback", "generate_hd_thumbnail")
    builder.add_edge("generate_hd_thumbnail", END)

    return builder.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Session State Init
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "step": "idle",
        "thread_id": str(uuid4()),
        "word_input": "",
        "thumbnails": [],
        "chosen": 1,
        "feedback": "",
        "final_thumbnail": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session():
    for key in ["step", "thread_id", "word_input", "thumbnails", "chosen", "feedback", "final_thumbnail"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Thumbnail Maker Agent")
st.caption("영단어 설명을 입력하면 YouTube Shorts 썸네일 3개를 생성합니다.")

init_session()
graph = compile_graph()
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ── Step 1: 입력 ────────────────────────────────────────────────────────────
if st.session_state.step == "idle":
    with st.form("input_form"):
        word_input = st.text_area(
            "영단어 아무거나! 설명도 있으면 좋아요!",
            placeholder="예: preview는 'pre(미리)'와 'view(본다)'가 합쳐졌으니 '미리 본다'는 뜻이다",
            height=150,
        )
        submitted = st.form_submit_button("썸네일 생성 시작")

    if submitted and word_input.strip():
        st.session_state.word_input = word_input
        st.session_state.step = "generating"
        st.rerun()

# ── Step 2: 썸네일 5개 생성 ─────────────────────────────────────────────────
elif st.session_state.step == "generating":
    with st.spinner("3개의 썸네일을 생성하고 있습니다... (약 1~2분 소요)"):
        graph.invoke({"word_promt": st.session_state.word_input}, config=config)

    graph_state = graph.get_state(config)
    st.session_state.thumbnails = graph_state.values.get("thumbnail_sketches", [])
    st.session_state.step = "awaiting_feedback"
    st.rerun()

# ── Step 3: 썸네일 선택 + 피드백 입력 ──────────────────────────────────────
elif st.session_state.step == "awaiting_feedback":
    st.subheader("생성된 썸네일 — 마음에 드는 것을 선택하세요")
    cols = st.columns(3)
    for i, img_bytes in enumerate(st.session_state.thumbnails):
        with cols[i]:
            st.image(img_bytes, caption=f"#{i + 1}", use_container_width=True)

    with st.form("feedback_form"):
        chosen = st.radio(
            "썸네일 번호",
            options=[1, 2, 3],
            horizontal=True,
        )
        feedback = st.text_area(
            "수정 요청사항을 말씀해주세요! (없어도 괜찮아요)",
            placeholder="예: 로고를 제거하고, 사실적인 3D 스타일로 만들어줘.",
            height=100,
        )
        submitted = st.form_submit_button("HD 썸네일 생성")

    if submitted:
        st.session_state.chosen = chosen
        st.session_state.feedback = feedback
        st.session_state.step = "generating_hd"
        st.rerun()

# ── Step 4: HD 썸네일 생성 ──────────────────────────────────────────────────
elif st.session_state.step == "generating_hd":
    with st.spinner("HD 썸네일을 생성하고 있습니다..."):
        graph.invoke(
            Command(resume={
                "user_feedback": st.session_state.feedback,
                "chosen_prompt": st.session_state.chosen,
            }),
            config=config,
        )

    graph_state = graph.get_state(config)
    st.session_state.final_thumbnail = graph_state.values.get("final_thumbnail")
    st.session_state.step = "done"
    st.rerun()

# ── Step 5: 결과 ────────────────────────────────────────────────────────────
elif st.session_state.step == "done":
    st.subheader("최종 HD 썸네일")
    if st.session_state.final_thumbnail:
        st.image(st.session_state.final_thumbnail, use_container_width=True)
        st.download_button(
            label="썸네일 다운로드",
            data=st.session_state.final_thumbnail,
            file_name="thumbnail_final.jpg",
            mime="image/jpeg",
        )

    if st.button("처음부터 다시 시작"):
        reset_session()
