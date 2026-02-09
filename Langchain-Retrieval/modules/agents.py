import json
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import operator
from langchain_core.messages import AIMessage
from typing import TypedDict, Annotated, List


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)  # node 1 -> LLM thinking
        graph.add_node("action", self.take_action)  # node 2 -> eksekusi tools

        # condional edge, True (llm -> action), false (LLM -> END)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")  # edge action -> llm (balik lagi ke llm)
        graph.set_entry_point("llm")  # node pintu masuk -> llm

        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]

        if not isinstance(result, AIMessage):
            return False

        if len(result.tool_calls) > 0:
            return True

        # cek apakah ada json dari hasil jawaban AI, if True need action or tools
        content = result.content if result.content else ""
        if '[{"name":' in content or "tool_calls" in content:
            print("🕵️ Terdeteksi JSON Tool Call di dalam teks!")
            return True

        return False

    def call_model(self, state: AgentState):
        messages = state["messages"]

        if len(messages) > 10:
            return {
                "messages": [
                    AIMessage(
                        content="Maaf, saya mencoba mencari tapi prosesnya terlalu lama (Loop detected). Mohon perjelas pertanyaan."
                    )
                ]
            }

        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        try:
            # Tambahkan print debug
            print("🤖 Model sedang berpikir...")
            response = self.model.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"Error invoke: {e}")
            return {"messages": [AIMessage(content="Error system.")]}

    def take_action(self, state: AgentState):
        last_message = state["messages"][-1]
        results = []
        tool_calls = []

        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            tool_calls = last_message.tool_calls

        # manual parsing atau lihat tools yang dibutuhkan
        else:
            try:
                content = last_message.content
                # Cari posisi kurung siku JSON [...]
                start_idx = content.find('[{"name":')
                if start_idx != -1:
                    json_str = content[start_idx:]
                    # Bersihkan jika ada sisa text di belakang (opsional)
                    end_idx = json_str.rfind("}]") + 2
                    json_str = json_str[:end_idx]

                    parsed_tools = json.loads(json_str)

                    # Konversi ke format standard tool call
                    for pt in parsed_tools:
                        tool_calls.append(
                            {
                                "name": pt["name"],
                                "args": pt["arguments"],
                                "id": "manual_call",  # ID dummy
                            }
                        )
            except Exception as e:
                print(f"❌ Gagal parsing manual JSON: {e}")

        # EKSEKUSI TOOLS
        for t in tool_calls:
            print(f"🛠️ Eksekusi Tool: {t['name']} dengan args: {t['args']}")

            if t["name"] not in self.tools:
                result = "Error: Tool name not found. Please check valid tools."
            else:
                try:
                    # Pastikan args adalah dict
                    args = t["args"]
                    if isinstance(args, str):
                        args = json.loads(args)

                    result = self.tools[t["name"]].invoke(args)

                    # Jika hasil kosong, beri tahu model secara eksplisit
                    if not result:
                        result = "Info: Data tidak ditemukan di database untuk input tersebut."

                except Exception as e:
                    result = f"Error execution: {str(e)}"

            print(f"   📄 Hasil Tool: {str(result)[:100]}...")

            results.append(
                ToolMessage(
                    tool_call_id=t.get("id", "manual_call"),
                    name=t["name"],
                    content=str(result),
                )
            )

        print("🔙 Kembali ke Model membawa data...")
        return {"messages": results}
