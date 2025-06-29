from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 导入OpenAI相关模型
from dotenv import load_dotenv, find_dotenv  # 导入dotenv用于加载环境变量
import os  # 导入os用于环境变量读取
from langchain import hub  # 导入hub用于拉取预设prompt
from langchain_community.tools.tavily_search import TavilySearchResults  # 导入Tavily搜索工具
from langchain.agents import create_tool_calling_agent  # 创建工具调用智能体
from langchain.agents import AgentExecutor  # 智能体执行器
from langchain_community.chat_message_histories import ChatMessageHistory  # 聊天消息历史管理
from langchain_core.runnables.history import RunnableWithMessageHistory  # 可带历史的可运行对象
import json  # 导入json用于历史消息的持久化
import gradio as gr  # 导入gradio用于Web界面
from langchain_community.document_loaders import WebBaseLoader  # 网页文档加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割器
from langchain_community.vectorstores import FAISS  # 向量数据库FAISS

load_dotenv(find_dotenv())  # 加载.env环境变量

# 全局消息历史存储字典，key为session_id，value为ChatMessageHistory对象
store = {}

# 加载指定session_id的历史消息到store
# 如果历史文件不存在则新建空历史
# 支持content/text两种字段
# 并根据type区分human/ai消息

def load_history(session_id):
    try:
        with open(f"history_{session_id}.json", "r", encoding="utf-8") as f:
            messages = json.load(f)
        store[session_id] = ChatMessageHistory()
        for msg in messages:
            content = msg.get("content", msg.get("text", ""))
            if msg["type"] == "human":
                store[session_id].add_user_message(content)
            else:
                store[session_id].add_ai_message(content)
    except FileNotFoundError:
        store[session_id] = ChatMessageHistory()

# 获取指定session_id的历史对象，不存在则自动加载

def get_session_history(session_id):
    if session_id not in store:
        load_history(session_id)
    return store[session_id]

# 保存指定session_id的历史消息到本地json文件

def save_history(session_id):
    history = [msg.to_json() for msg in store[session_id].messages]
    with open(f"history_{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 初始化全局模型和智能体
model = ChatOpenAI(
    model="gpt-3.5-turbo",  # 使用的OpenAI模型
    api_key=os.environ['OPENAI_API_KEY'],  # API密钥
    base_url=os.environ['OPENAI_BASE_URL'],  # API地址
)
prompt = hub.pull('hwchase17/openai-functions-agent')  # 拉取预设prompt
search = TavilySearchResults(max_results=5)  # 初始化Tavily搜索工具
tools = [search]  # 工具列表
agent = create_tool_calling_agent(model, tools, prompt)  # 创建工具调用智能体
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # 智能体执行器
hist_agent = RunnableWithMessageHistory(
    runnable=executor,  # 代理执行器
    get_session_history=get_session_history,  # 获取历史的回调
    input_messages_key="input",  # 输入消息的key
    history_messages_key="history",  # 历史消息的key
)

# 聊天主函数，处理一次对话并返回历史消息格式
# 返回值为gradio可用的history格式

def chat_fn(user_input, session_id):
    response = hist_agent.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    save_history(session_id)  # 保存历史
    history_msgs = response.get("history", [])  # 获取历史消息
    # 转换为gr.Chatbot需要的格式
    history = [(msg.content, "") if msg.type == "human" else ("", msg.content) for msg in history_msgs]
    # 添加当前对话
    ai_message = response.get("output", "")  # 获取AI的回复
    history.append((user_input, ai_message))  # 添加当前对话到历史
    return history, gr.update(value="")

FAISS_DB_PATH = "faiss_index"  # FAISS数据库路径

# 从指定url构建FAISS知识库
def build_faiss_from_url(url, db_path=FAISS_DB_PATH):
    loader = WebBaseLoader(url)  # 加载网页内容
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)  # 文本分割
    splits = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_BASE_URL'],
    )
    vectorstore = FAISS.from_documents(splits, embeddings)  # 构建向量库
    FAISS.save_local(db_path, vectorstore)  # 使用save_local方法保存向量库
    return vectorstore

# 加载本地FAISS知识库
def load_faiss(db_path=FAISS_DB_PATH):
    embeddings = OpenAIEmbeddings(
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_BASE_URL'],
    )
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings)  # 使用load_local方法加载向量库
    else:
        return None

vectorstore = load_faiss()  # 全局知识库对象

# 检索知识库内容

def search_knowledge(query, vectorstore, top_k=3):
    if vectorstore is None:
        return ["知识库未加载"]
    docs_and_scores = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs_and_scores]

# Gradio界面搭建
with gr.Blocks() as demo:
    gr.Markdown("# 多会话智能助手")  # 标题
    session_id = gr.Textbox(label="Session ID", value="default_session_id")  # 会话ID输入框
    chatbot = gr.Chatbot(label="对话历史")  # 聊天历史展示
    user_input = gr.Textbox(label="用户输入")  # 用户输入框
    send_btn = gr.Button("发送")  # 发送按钮

    # 发送按钮回调函数，调用chat_fn并返回历史
    def on_send(user_input, session_id, chat_history):
        history, _ = chat_fn(user_input, session_id)
        return history, ""  # 返回历史和清空用户输入
    send_btn.click(
        on_send,
        inputs=[user_input, session_id, chatbot],
        outputs=[chatbot, user_input]  # 添加user_input作为输出
    )
    user_input.submit(
        on_send,
        inputs=[user_input, session_id, chatbot],
        outputs=[chatbot, user_input]  # 添加user_input作为输出
    )

if __name__ == "__main__":
    demo.launch()  # 启动Gradio界面
    