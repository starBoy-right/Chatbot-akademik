from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from agent_blocks import llm_model
from agent_blocks import embeddings_model
from agent_blocks import vector_store
from typing import cast

import chainlit as cl



@cl.on_chat_start
async def on_chat_start():
    model = llm_model
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions. And always answer in bahasa indonesia",
            ),
            ("human", "{question}")
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()