from langchain_openai import ChatOpenAI  # 导入 OpenAI 聊天模型的接口
from dotenv import load_dotenv, find_dotenv  # 导入 dotenv 用于加载环境变量
import os  # 导入 os 用于读取环境变量
from langchain import hub  # 导入 hub 用于拉取预设的 prompt
from langchain_community.tools.tavily_search import TavilySearchResults  # 导入 Tavily 搜索工具
from langchain.agents import create_tool_calling_agent  # 导入创建工具调用代理的函数
from langchain.agents import AgentExecutor  # 导入代理执行器
from langchain_openai import OpenAIEmbeddings  # 导入 OpenAI 向量嵌入模型
from langchain_community.vectorstores import FAISS  # 导入 FAISS 向量数据库
from langchain.memory import ChatMessageHistory  # 导入对话消息历史管理类
from langchain.schema.runnable import RunnableWithMessageHistory  # 导入带消息历史的可运行对象

from langchain_community.document_loaders import WebBaseLoader  # 导入网页文档加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 导入递归文本分割器
from langchain.tools.retriever import create_retriever_tool  # 导入检索工具创建函数

if __name__ == "__main__":
    load_dotenv(find_dotenv())  # 加载 .env 文件中的环境变量

    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"]
    )  # 初始化 OpenAI 聊天模型

    prompt = hub.pull("hwchase17/openai-functions-agent")  # 拉取预设 prompt

    search = TavilySearchResults(max_results=5)  # Tavily 搜索工具
    search.invoke("question")  # 测试搜索工具

    # loader = WebBaseLoader("https://lilianweng.github.io/posts/2025-05-01-thinking/#thinking-as-latent-variables")  # 加载网页内容
    # documents = loader.load()  # 加载文档

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500,
    #     chunk_overlap=150,
    #     separators=["\n", " ", ""]
    # )  # 文本分割器
    # docs = text_splitter.split_documents(documents)  # 分割文档
    # print(f"分割后的文档数量: {len(docs)}")  # 输出分割后文档数量

    # embedding_model = OpenAIEmbeddings(
    #     model="text-embedding-3-small",
    #     api_key=os.environ["OPENAI_API_KEY"],
    #     base_url=os.environ["OPENAI_BASE_URL"]
    # )  # 嵌入模型

    # vector = FAISS.from_documents(docs, embedding_model)  # 构建向量数据库
    # retriever = vector.as_retriever()  # 获取检索器
    # retriever_tool = create_retriever_tool(
    #     retriever,
    #     name="web_blog_retriever",
    #     description="探讨测试时计算(如思维链CoT、并行采样、顺序修订、强化学习 RL)对提升语言模型推理能力的作用，分析其与人类思维的类比、潜在变量建、外部工具使用及忠实性挑战，指出优化测试时计算可作为模型性能提升的新维度，且需平衡计算成本与推理效果。")  # 创建检索工具

    tools = [search, ]  # 工具列表
    agent = create_tool_calling_agent(model, tools, prompt)  # 创建智能体
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # 创建代理执行器

    store = {}  # 用于存储不同 session 的对话历史
    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()  # 新建对话历史
        return store[session_id]  # 返回对应 session 的历史
    
    hist_agent = RunnableWithMessageHistory(
        runnable=executor,  # 代理执行器
        get_session_history=get_session_history,  # 获取历史的回调
        input_messages_key="input",  # 输入消息的 key
        history_messages_key="history",  # 历史消息的 key
    )  # 包装成带历史的智能体

    while True:
        user_input = input("请输入您的问题：")
        response = executor.invoke({"input": user_input})
        print(response)
    