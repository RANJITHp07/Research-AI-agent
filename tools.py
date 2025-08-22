from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool

# Initialize DuckDuckGo search tool
duck_duck_search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=duck_duck_search.run,
    description=(
        "Use this tool to search the web for up-to-date information, "
        "fact-check answers, fill knowledge gaps, and retrieve live data "
        "that the AI model may not have. Ideal for current events, recent news, "
        "or any query requiring external, real-time information."
    )
)

wiki_api = WikipediaAPIWrapper()

# Initialize Wikipedia query tool with the API wrapper
wikipedia_query = WikipediaQueryRun(api_wrapper=wiki_api)
wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia_query.run,
    description=(
        "Use this tool to retrieve detailed and well-structured information "
        "from Wikipedia articles. Ideal for background knowledge, explanations, "
        "historical context, and factual data verified by Wikipedia."
    )
)

tools = [search_tool, wikipedia_tool]
