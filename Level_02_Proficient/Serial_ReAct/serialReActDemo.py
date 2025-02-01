# coding=utf-8
"""
@Author: Jacob Y
@Date  : 1/31/2025
@Desc  : OpenAI 串行调用多个工具的 LLM 逻辑
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


if __name__ == "__main__":
    react_llm = ReactLLM()
    result = react_llm.chat(
        "我很开心，请推荐",
        {
            "mood_based_dish": mood_based_dish,
            "dish_and_drink_pairing": dish_and_drink_pairing,
            "drink_and_music_pairing": drink_and_music_pairing,
            "music_and_activity_suggestion": music_and_activity_suggestion,
            "activity_and_travel_recommendation": activity_and_travel_recommendation,
        },
    )
    rm = re.search(r'(?:Answer: )(.*)', result)
    print(rm.group(1))
    # 既然您感到开心，我为您推荐了一顿美食：“辣炒鸡丁”，搭配一杯“龙舌兰日出”鸡尾酒。为增添气氛，可以听一曲“阳光海岸”。
    # 在这样的音乐伴随下，进行一场海滩排球也是极好的。为此，我推荐您考虑旅行到马尔代夫，完美结束这段愉快的体验。

"""
Thought: 用户心情是开心，我需要根据他的心情推荐一个菜品。
Action: 调用 `mood_based_dish`
Input: mood="开心"
Tool: 推荐菜品为“辣炒鸡丁”。

Thought: 现在我有了推荐的菜品，接下来需要为这道菜推荐一个搭配的鸡尾酒。
Action: 调用 `dish_and_drink_pairing`
Input: dish="辣炒鸡丁"
Tool: 推荐鸡尾酒为“龙舌兰日出”。

Thought: 我已经得到了鸡尾酒的推荐，现在需找到与此鸡尾酒搭配的音乐。
Action: 调用 `drink_and_music_pairing`
Input: drink="龙舌兰日出"
Tool: 推荐音乐为“热带之家”。

Thought: 已经有了音乐建议，我需要根据这首音乐推荐一个适合的活动。
Action: 调用 `music_and_activity_suggestion`
Input: music="热带之家"
Tool: 推荐活动为“海滩派对”。

Thought: 最后一步是根据推荐的活动推荐旅行目的地。
Action: 调用 `activity_and_travel_recommendation`
Input: activity="海滩派对"
Tool: 推荐旅行目的地为“马尔代夫”。

Final
Answer: 当你感到开心时，你可以享用一道辣炒鸡丁，搭配一杯龙舌兰日出鸡尾酒，同时听着“热带之家”的音乐，参加一个海滩派对。为此，你可以选择马尔代夫作为你的旅行目的地，完美地体验这一切乐趣！
"""