#!/usr/bin/env python3
"""
简单的调试测试脚本
用于在VS Code中设置断点和调试LangGraph项目
"""

import os
import sys
from pathlib import Path

# 添加src目录到Python路径
backend_dir = Path(__file__).parent
src_dir = backend_dir / "src"
sys.path.insert(0, str(src_dir))

from langchain_core.messages import HumanMessage
from agent.graph import graph


def debug_simple_query():
    """调试简单查询功能"""
    print("🔍 开始调试简单查询...")
    
    # 在这里设置断点！
    state = {
        "messages": [HumanMessage(content="拜登和特朗普在移民政策上的差异体现在哪些方面，这些差异会对移民现象产生什么样的影响")],
        "initial_search_query_count": 2,
        "max_research_loops": 1,
    }
    
    print(f"📝 输入状态: {state}")
    
    # 在这里设置断点查看graph.invoke的执行过程
    try:
        result = graph.invoke(state)
        print("✅ 查询成功完成!")
        
        # 在这里设置断点查看结果
        messages = result.get("messages", [])
        if messages:
            print(f"📋 最终回答: {messages[-1].content[:200]}...")
        else:
            print("❌ 没有收到回答")
            
        print(f"🔗 收集的源: {len(result.get('sources_gathered', []))}个")
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        raise


def debug_configuration():
    """调试配置加载"""
    print("⚙️ 调试配置加载...")
    
    from agent.configuration import Configuration
    
    # 设置断点查看配置加载过程
    config = Configuration.from_runnable_config(None)
    
    print(f"🤖 查询生成模型: {config.query_generator_model}")
    print(f"🤔 反思模型: {config.reflection_model}")
    print(f"✍️ 答案模型: {config.answer_model}")
    print(f"🔢 初始查询数: {config.number_of_initial_queries}")
    print(f"🔄 最大循环数: {config.max_research_loops}")


def debug_state_types():
    """调试状态类型"""
    print("📊 调试状态类型...")
    
    from agent.state import OverallState, QueryGenerationState
    
    # 设置断点查看状态类型
    sample_state: OverallState = {
        "messages": [HumanMessage(content="测试消息")],
        "search_query": ["查询1", "查询2"],
        "web_research_result": ["结果1"],
        "sources_gathered": [],
        "initial_search_query_count": 2,
        "max_research_loops": 1,
        "research_loop_count": 0,
        "reasoning_model": "gemini-2.5-flash"
    }
    
    print(f"状态键: {list(sample_state.keys())}")


def main():
    """主调试函数"""
    print("🚀 启动LangGraph项目调试")
    print("=" * 50)
    
    # 检查环境变量
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置GEMINI_API_KEY环境变量")
        return
    
    print(f"✅ GEMINI_API_KEY已设置 (长度: {len(api_key)})")
    
    try:
        # 1. 调试配置
        debug_configuration()
        print("-" * 30)
        
        # 2. 调试状态类型
        debug_state_types() 
        print("-" * 30)
        
        # 3. 调试简单查询 (在这里设置主要断点)
        debug_simple_query()
        
    except Exception as e:
        print(f"❌ 调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
