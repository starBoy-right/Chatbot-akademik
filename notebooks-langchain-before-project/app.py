from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from agent_blocks import llm_model
from agent_blocks import setup_and_retriev
from typing import cast

import chainlit as cl



@cl.on_chat_start
async def on_chat_start():
    model = llm_model
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Kamu adalah NufiChat, asisten cerdas buatan Romi untuk tugas akhir.
            Tugasmu adalah menjawab pertanyaan berdasarkan diskusi grup akademik.
            
            INSTRUKSI:
            1. Jika pengguna menyapa (misal: "Halo", "Selamat Pagi", "Siapa kamu?"), jawablah dengan ramah dan perkenalkan diri secara natural TANPA melihat konteks.
            2. Jika pengguna bertanya tentang data/informasi/diskusi, JAWABLAH HANYA berdasarkan KONTEKS berikut ini.
            
            KONTEKS:
            {context}

            3. Jika jawaban tidak ada di dalam konteks, katakan "Maaf, informasi tidak ditemukan dalam database."
            """
        ),
        ("human", "{question}")
    ]
)
    
    runnable = setup_and_retriev | prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    res = await runnable.ainvoke(
        message.content,
        config={"callbacks": [cl.LangchainCallbackHandler()]}
    )

    await cl.Message(content=res).send()