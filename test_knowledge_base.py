from langchain_tool_hist_gradio import build_faiss_from_url, load_faiss, search_knowledge
from dotenv import load_dotenv, find_dotenv
import os

def test_knowledge_base():
    # 1. 加载环境变量
    load_dotenv(find_dotenv())
    
    # 2. 测试构建知识库
    # 使用LangChain的文档页面作为测试
    url = "https://python.langchain.com/docs/get_started/introduction"
    print("开始从URL构建知识库...")
    vectorstore = build_faiss_from_url(url)
    print("知识库构建完成！")
    
    # 3. 测试加载知识库
    print("\n开始加载知识库...")
    loaded_vectorstore = load_faiss()
    if loaded_vectorstore is not None:
        print("知识库加载成功！")
    else:
        print("知识库加载失败！")
    
    # 4. 测试搜索功能
    test_queries = [
        "What is LangChain?",
        "How to use LangChain with LLMs?",
        "What are the main components of LangChain?"
    ]
    
    print("\n开始测试搜索功能...")
    for query in test_queries:
        print(f"\n查询: {query}")
        results = search_knowledge(query, loaded_vectorstore)
        print("搜索结果:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}\n")

if __name__ == "__main__":
    test_knowledge_base() 