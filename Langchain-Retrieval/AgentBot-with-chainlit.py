from modules.agents import Agent
from modules.tools import query_from_academic_rule, get_student_academic_record
from modules import blocks 
import chainlit as cl
import os
from dotenv import load_dotenv

load_dotenv()

# 2. (Opsional) Debugging: Cek apakah env var sudah terbaca dengan benar
if os.environ.get("LANGSMITH_TRACING") == "true":
    print("✅ LangSmith Tracing is ON")
else:
    print("❌ LangSmith Tracing is OFF. Check your .env file")


@cl.on_chat_start
async def on_chat_start():
    llm_model = blocks.model
    prompt = blocks.system_prompt
    bot = Agent(
        model=llm_model,
        tools=[query_from_academic_rule, get_student_academic_record],
        system=prompt,
    )

    cl.user_session.set("agent", bot)
    print("Agent Ready")


@cl.on_message
async def on_message(message: cl.Message):
    bot = cl.user_session.get("agent")
    initial_step = {"messages": [("user", message.content)]}
    res = await cl.make_async(bot.graph.invoke)(initial_step)
    last_message = res["messages"][-1]
    await cl.Message(content=last_message.content).send()
