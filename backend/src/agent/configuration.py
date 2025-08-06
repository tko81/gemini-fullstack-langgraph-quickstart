import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""
    # 不同的功能使用不同的模型
    query_generator_model: str = Field(
        default="gemini-2.0-flash",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="gemini-2.5-flash",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="gemini-2.5-pro",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    # classmethod装饰器的方法 可以直接通过类调用，不需要先创建实例 类似于静态方法
    # 从多个数据源创建一个 Configuration 对象，数据源的优先级是：
    # 环境变量（最高优先级）
    # RunnableConfig 中的配置
    # Field 中的默认值（最低优先级）
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            # 先查环境变量（转为大写） 如果环境变量不存在，则查 configurable
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        # 只保留raw_values中的 非 None 的值 保存到 values 中
        values = {k: v for k, v in raw_values.items() if v is not None}

        # 最后创建 Configuration 实例 这里的 cls 是 Configuration 类本身
        # cls(**values) 会调用 Configuration 的构造函数 并传入 values 中的键值对作为参数
        # 这样就可以创建一个新的 Configuration 实例
        # **关键字参数解包 将字典"拆开"成关键字参数
        return cls(**values)
