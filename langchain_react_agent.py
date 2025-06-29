import os
from langchain.agents import create_react_agent, AgentExecutor  # 导入ReAct智能体相关方法
from langchain_community.agent_toolkits.load_tools import load_tools  # 导入工具加载方法
from langchain_community.chat_models import ChatZhipuAI  # 导入智谱AI聊天模型
from langchain_core.prompts import PromptTemplate  # 导入提示词模板
from dotenv import load_dotenv, find_dotenv  # 导入dotenv用于加载环境变量
from langchain.agents import create_react_agent  # 再次导入ReAct智能体方法（可合并）
from langchain.agents import AgentExecutor  # 再次导入智能体执行器（可合并）


if __name__ == "__main__":
    load_dotenv(find_dotenv())  # 加载.env环境变量
    model = ChatZhipuAI(model="glm-z1-air",
    api_key=os.environ["ZHIPUAI_API_KEY"])  # 初始化智谱AI模型，使用环境变量中的API KEY
    tools = load_tools(["serpapi", "llm-math"], llm=model)  # 加载serpapi和llm-math工具
    print(tools)  # 打印加载的工具列表

    # 提示词模板，指导智能体如何思考和输出
    template="""
    尽你所能使用中文回答以下问题，能够使以下工具解决问题：

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(template)  # 构建提示词模板
    agent = create_react_agent(llm=model, tools=tools,prompt=prompt)  # 创建ReAct智能体
    executor = AgentExecutor(
        agent=agent,  # 智能体对象
        tools=tools,  # 工具列表
        verbose=True,  # 输出详细日志
        handle_parsing_errors=True  # 容错参数（应为handle_parsing_errors，疑似拼写错误）
    )
    result = executor.invoke({"input":"查询2024年市场玫瑰花进货价格，我想知道如果加价5%的价格"})  # 执行一次智能体推理
    print(result['output'])  # 打印最终输出结果

