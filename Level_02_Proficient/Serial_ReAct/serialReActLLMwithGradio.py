# coding=utf-8
"""
@Author: Jacob Y
@Date  : 1/31/2025
@Desc  : 基于Gradio可视化UI的OpenAI 串行调用多个工具的LLM 逻辑
"""
import re
import json
import yaml
from openai import OpenAI
from openai_cli import get_client
from tools import (
    mood_based_dish,
    dish_and_drink_pairing,
    drink_and_music_pairing,
    music_and_activity_suggestion,
    activity_and_travel_recommendation,
)
import gradio as gr

class ReactLLM:
    def __init__(self) -> None:
        self.system_prompt = """
        你是一个智能助手，你需要按照严格的顺序调用以下工具：
        1. `mood_based_dish` —— 根据用户输入的心情推荐菜品
        2. `dish_and_drink_pairing` —— 根据推荐的菜品推荐搭配的鸡尾酒
        3. `drink_and_music_pairing` —— 根据推荐的鸡尾酒推荐适合的音乐
        4. `music_and_activity_suggestion` —— 根据推荐的音乐推荐适合的活动
        5. `activity_and_travel_recommendation` —— 根据推荐的活动推荐适合的旅行目的地

        你的回答必须按照以下格式：
        
        Thought: 你在思考下一步该做什么
        Action: 需要调用的工具名称
        Action
        Input: 该工具的输入参数
        tool: 该工具返回的结果
        ...(重复
        Thought / Action / Action
        Input / tool)
        Final
        Answer: 根据所有工具的结果，生成最终的完整回答

        确保你按 ** 严格顺序 ** 串行调用这些工具，不要跳过任何步骤。
        """

        self.client = get_client()
        # self.model_type = self.client.models.list().data[0].id
        self.model_type = "gpt-4-turbo"
        self.tools = {}

    def _load_tools_yaml_file(self, file_path):
        """加载 YAML 配置"""
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        self.system_prompt = self.system_prompt.format(tools=data)
        return data

    def _load_tools(self, tools: dict):
        """注册工具"""
        self.tools = tools

    def _call_tool(self, tool_call):
        """调用 **单个** 工具"""
        tool_name = tool_call["name"]
        params = tool_call["parameters"]

        if tool_name in self.tools:
            return {tool_name: self.tools[tool_name](**params)}
        else:
            return {tool_name: {"error": "工具未找到"}}

    def _call_model(self, messages):
        """调用 OpenAI API"""
        resp = self.client.chat.completions.create(
            model=self.model_type,
            messages=messages,
            max_tokens=4096,
            # tool_choice="auto",
            seed=42,
        )

        return resp.choices[0].message

    def chat(self, query, tools):
        """主流程 - 逐步串行调用工具"""
        self._load_tools_yaml_file("functions.yaml")
        self._load_tools(tools)

        history = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": query}]

        while True:
            # 让 LLM 进行推理
            assistant_response = self._call_model(history)
            history.append({"role": "assistant", "content": assistant_response.content})

            # 解析工具调用
            if assistant_response.tool_calls:
                # 只处理 **第一个** 工具（串行调用）
                tool_call = assistant_response.tool_calls[0]
                tool_response = self._call_tool(tool_call)

                # 记录工具返回结果
                history.append(
                    {
                        "role": "tool",
                        "name": tool_call["name"],
                        "content": json.dumps(tool_response, ensure_ascii=False),
                    }
                )
            else:
                # 结束循环
                break

        return assistant_response.content


# Gradio界面部分（带按钮）
def gradio_interface(query):
    tools = {
            "mood_based_dish": mood_based_dish,
            "dish_and_drink_pairing": dish_and_drink_pairing,
            "drink_and_music_pairing": drink_and_music_pairing,
            "music_and_activity_suggestion": music_and_activity_suggestion,
            "activity_and_travel_recommendation": activity_and_travel_recommendation,
        }
    react_llm = ReactLLM()
    response = react_llm.chat(query, tools)

    match = re.search(r'Answer:\s*(.*)', response, re.S)
    return match.group(1).strip() if match else "未找到合适的推荐结果"


# 创建Gradio界面（带按钮）
def run_chat(query):
    return gradio_interface(query)




if __name__ == "__main__":
    with gr.Blocks(title="MoodFlow AI", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
            <div style="text-align: center;">
                <h1>🌊 MoodFlow AI 🎶🍹</h1>
                <h3>让你的心情，变成一场完美的体验</h3>
                <h3>推荐内容:食物、饮品、音乐、活动、旅行</h3>
            </div>
            """)
        with gr.Row():
            mood_input = gr.Textbox(label="你的心情", placeholder="输入你的心情，比如 放松/疲惫/兴奋/难过/无聊")
            output = gr.Textbox(label="MoodFlow 推荐", interactive=False)
        submit_btn = gr.Button("生成推荐", variant="primary")
        submit_btn.click(fn=run_chat, inputs=mood_input, outputs=output)

    demo.launch()
