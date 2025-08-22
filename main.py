from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import tools
import json
import re



load_dotenv()

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources:list[str]
    tools_used:list[str]

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    parser= PydanticOutputParser(pydantic_object=ResearchResponse)

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

    agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt, 
    tools=tools
    )

    agent_executor=AgentExecutor(
    agent=agent,
    tools=tools, 
    verbose=True,
    max_iterations=3,
    max_execution_time=30,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    early_stopping_method="generate"
    )

    try:
        query=input("What can I help you to research about:\n")
        raw_response=agent_executor.invoke({"query":query})
    except Exception as error:
        print ("AI assistant is busy",error) 

    print(raw_response)   

    output_text = raw_response['output']

    print(output_text)

    if not output_text.strip():
         print("No output from the agent")
    else:
            try:
               json_str = re.sub(r"^```json\s*|\s*```$", "", output_text, flags=re.DOTALL)
               parsed_output = json.loads(json_str)
               print(parsed_output)     
            except Exception as error:
                print ("AI assistant is busy",error)     
except Exception as error:
    print ("Error is",error)
