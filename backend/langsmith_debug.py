#!/usr/bin/env python3
"""
LangSmith 可视化调试脚本
专门用于在 LangSmith 中查看 LangGraph 流程图
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 添加src目录到Python路径
backend_dir = Path(__file__).parent
src_dir = backend_dir / "src"
sys.path.insert(0, str(src_dir))

from langchain_core.messages import HumanMessage
from agent.graph import graph


def run_with_langsmith_tracing(query: str, run_name: str = None):
    """使用 LangSmith 追踪运行查询"""
    
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"research_query_{timestamp}"
    
    print("🔍 LangSmith 追踪查询")
    print("=" * 50)
    print(f"📊 运行名称: {run_name}")
    print(f"❓ 查询内容: {query}")
    print()
    
    # 配置状态
    state = {
        "messages": [HumanMessage(content=query)],
        "initial_search_query_count": 2,  # 减少查询数量以便更好地观察
        "max_research_loops": 1,          # 限制循环次数
    }
    
    try:
        # 使用 LangSmith 追踪配置
        config = {
            "run_name": run_name,
            "tags": ["debug", "langsmith", "research-agent"],
            "metadata": {
                "environment": "development",
                "query_type": "research",
                "version": "1.0"
            }
        }
        
        print("🚀 开始执行 LangGraph...")
        result = graph.invoke(state, config=config)
        
        print("✅ 执行完成!")
        print(f"📋 最终回答长度: {len(result.get('messages', [])[0].content) if result.get('messages') else 0} 字符")
        print(f"🔗 收集的源数量: {len(result.get('sources_gathered', []))} 个")
        
        print("\n" + "=" * 60)
        print("🌐 LangSmith 可视化指南:")
        print("=" * 60)
        print("1. 访问: https://smith.langchain.com")
        print("2. 登录你的账号")
        print("3. 选择项目: 'gemini-research-agent'")
        print(f"4. 查找运行: '{run_name}'")
        print("5. 点击运行记录进入详情页面")
        print()
        print("📊 在详情页面你可以看到:")
        print("   • Graph View: 完整的流程图")
        print("   • Trace View: 详细的执行追踪")
        print("   • Timeline: 时间轴视图")
        print("   • Inputs/Outputs: 每个节点的输入输出")
        print("   • Performance: 性能分析")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        raise


def main():
    """主函数"""
    
    # 检查环境变量
    required_env_vars = ["GEMINI_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2"]
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ 缺少环境变量:")
        for var in missing_vars:
            print(f"   • {var}")
        print("\n请确保在 .env 文件中设置以下变量:")
        print("GEMINI_API_KEY=your_gemini_key")
        print("LANGSMITH_API_KEY=your_langsmith_key")
        print("LANGCHAIN_TRACING_V2=true")
        print("LANGCHAIN_PROJECT=gemini-research-agent")
        return
    
    print("✅ 所有环境变量已设置")
    print(f"📊 LangSmith 项目: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print()
    
    # 示例查询
    queries = [
        "人工智能对教育的影响有哪些？",
        "气候变化的主要原因和解决方案",
        "区块链技术的实际应用场景"
    ]
    
    print("📝 可用的示例查询:")
    for i, query in enumerate(queries, 1):
        print(f"   {i}. {query}")
    print()
    
    # 运行第一个查询作为示例
    run_with_langsmith_tracing(queries[0], "ai_education_impact")


if __name__ == "__main__":
    main()
