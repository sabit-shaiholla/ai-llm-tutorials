import os
import asyncio
import datetime
import json
from typing import Any, Dict
from dataclasses import dataclass

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from loguru import logger
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.result import RunResult
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient
from dotenv import load_dotenv

load_dotenv()

model = GeminiModel('gemini-2.0-flash-exp', api_key=os.getenv('GEMINI_API_KEY'))
tavily_client = AsyncTavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

@dataclass
class SearchDataclass:
    max_results: int
    todays_date: str

@dataclass
class ResearchDependencies:
    todays_date: str

class ResearchResult(BaseModel):
    research_title: str = Field(description='Markdown heading describing the article topic, prefixed with #')
    research_main: str = Field(description='A main section that provides a detailed news article')
    research_bullets: str = Field(description='A set of bullet points summarizing key points')

search_agent = Agent(
    model,
    deps_type=ResearchDependencies,
    result_type=ResearchResult,
    system_prompt=(
        "You are a helpful research assistant and an expert in research. "
        "Given a single user query, you will call the 'get_search' tool exactly once, then combine the results. "
        "Return a JSON response that precisely matches the 'ResearchResult' fields: "
        "research_title, research_main, research_bullets. "
        "For example:\n"
        "{\n"
        "  \"research_title\": \"# My Article Title\",\n"
        "  \"research_main\": \"Full article text here...\",\n"
        "  \"research_bullets\": \"- Summary point\\n- Another point\\n\"\n"
        "}"
    ),
)

@search_agent.system_prompt
async def add_current_date(ctx: RunContext[ResearchDependencies]) -> str:
    return (
        f"You have today's date: {ctx.deps.todays_date}. "
        f"Please produce only one function call, then return your final answer as valid JSON for 'ResearchResult'."
    )

@search_agent.tool
async def get_search(search_data: RunContext[SearchDataclass], query: str) -> dict:
    """Perform a search using the Tavily client."""
    max_results = search_data.deps.max_results
    results = await tavily_client.get_search_context(
        query=query,
        max_results=max_results
    )
    logger.debug(f"Raw search results: {results}")
    return json.loads(results)

async def do_search(query: str, max_results: int):
    current_date = datetime.date.today()
    date_string = current_date.strftime("%Y-%m-%d")
    deps = SearchDataclass(max_results=max_results, todays_date=date_string)
    try:
        result = await search_agent.run(query, deps=deps)
        logger.debug(f"Search agent result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during search: {e}")
        return None

st.set_page_config(page_title = "AI News Researcher", page_icon = "🔍", layout = 'centered')
st.title("LLM News Researcher")
st.write("Welcome to the AI News Researcher. Enter a topic to get started.")

st.sidebar.title("Search Parameters")
query = st.sidebar.text_input("Enter your query:", value="latest Large Language Model news")
max_results = st.sidebar.slider("Number of search results:", min_value = 5, max_value = 20, value = 10)

st.write("Use sidebar to adjust search parameters.")

def handle_result_data(result_data: RunResult) -> None:
    """
    Validates and displays the research result.
    
    :param result_data: RunResult from the LLM/agent, expected to include ResearchResult in .data
    """
    if not result_data or not isinstance(result_data, RunResult) or not hasattr(result_data, 'data'):
        st.error("No valid research result was returned. Please try again.")
        logger.warning("The response from the model was invalid or incomplete.")
        return

    research_result = result_data.data
    if research_result is None:
        st.error("No valid research result was returned. Please try again.")
        logger.warning("The RunResult contains no data.")
        return

    display_research_result(research_result)
    st.success("Result displayed successfully!")

def display_research_result(research_result: ResearchResult) -> None:
    """
    Render the research result in Streamlit.

    :param research_result: A validated ResearchResult object containing
                            the title, main text, and bullet points.
    """
    st.markdown(research_result.research_title)
    st.markdown(
        f"<div style='line-height:1.6;'>{research_result.research_main}</div>",
        unsafe_allow_html=True
    )
    st.markdown("### Key Takeaways")
    bullet_points = research_result.research_bullets.strip().split("\\n")
    for point in bullet_points:
        if point.strip():
            st.markdown(f"- {point.strip()}")

if st.button("Get latest Large Language Model news"):
    with st.spinner("Searching for latest news..."):
        result_data = asyncio.run(do_search(query, max_results))
    
    logger.debug(f"Final result_data: {result_data}")
    
    if result_data is None:
        st.error("An error occurred during the search. Please try again.")
    else:
        handle_result_data(result_data)
