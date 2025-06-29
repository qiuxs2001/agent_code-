from langchain_openai import ChatOpenAI  # 导入OpenAI聊天模型
from langchain.agents import create_tool_calling_agent  # 创建工具调用智能体
from dotenv import load_dotenv, find_dotenv  # 加载环境变量
import os  # 读取环境变量
from langchain import hub  # 拉取预设prompt
from langchain_community.tools.tavily_search import TavilySearchResults  # Tavily搜索工具
from langchain.agents import AgentExecutor  # 智能体执行器
from langchain_core.runnables.history import RunnableWithMessageHistory  # 可带历史的可运行对象
from langchain_community.document_loaders import WebBaseLoader  # 网页文档加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割器
from langchain_community.vectorstores import FAISS  # 向量数据库FAISS
from langchain.tools.retriever import create_retriever_tool  # 检索工具
from langchain_community.chat_message_histories import ChatMessageHistory  # 聊天消息历史管理
import json  # 用于历史消息的持久化

# 保存指定session_id的历史消息到本地json文件

def save_history(session_id):
    history = [msg.to_json() for msg in store[session_id].messages]
    with open(f"history_{session_id}.json", "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 加载指定session_id的历史消息到store
# 如果历史文件不存在则新建空历史

def load_history(session_id):
    try:
        with open(f"history_{session_id}.json", "r") as f:
            messages = json.load(f)
        store[session_id] = ChatMessageHistory()
        store[session_id].parse_json(messages)
    except:
        store[session_id] = ChatMessageHistory()

if __name__ == "__main__":
    load_dotenv(find_dotenv())  # 加载.env环境变量

    model = ChatOpenAI(
        model="gpt-3.5-turbo",  # 使用的OpenAI模型
        api_key=os.environ["OPENAI_API_KEY"],  # API密钥
        base_url=os.environ["OPENAI_BASE_URL"],  # API地址
    )

    prompt = hub.pull("hwchase17/openai-functions-agent")  # 拉取预设prompt

    search = TavilySearchResults(max_results=5)  # 初始化Tavily搜索工具
    search.invoke("question")  # 测试搜索工具
    tools = [search, ]  # 工具列表
    agent = create_tool_calling_agent(model, tools, prompt)  # 创建工具调用智能体
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # 智能体执行器

    store = {}  # 全局消息历史存储字典，key为session_id，value为ChatMessageHistory对象

    # 获取指定session_id的历史对象，不存在则自动加载
    def get_session_history(session_id):
        if session_id not in store:
            load_history(session_id)
        return store[session_id]
    
    # 包装成带历史的智能体
    hist_agent = RunnableWithMessageHistory(
        runnable=executor,  # 代理执行器
        get_session_history=get_session_history,  # 获取历史的回调
        input_messages_key="input",  # 输入消息的key
        history_messages_key="history",  # 历史消息的key
    )

    # 主循环，持续接收用户输入并处理
    while True:
        user_input = input("输入您的问题：")  # 用户输入问题
        session_id = input("输入您的会话ID：")  # 用户输入会话ID

        response = hist_agent.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(response)  # 输出回复

        save_history(session_id)  # 保存历史
        print(f"当前session({session_id})的历史消息:")
        for msg in store[session_id].messages:
            print(msg)  # 打印历史消息