import asyncio
import os
import urllib.parse
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
from components.navigation import create_navigation
from components.styles import apply_global_styles

APP_TITLE = "智能助手"
APP_ICON = "🤖"

async def main() -> None:
    # 设置页面配置必须是第一个 Streamlit 命令
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # 应用全局样式
    apply_global_styles()
    
    # 创建导航
    create_navigation()
    
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "localhost")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("正在连接智能助手服务..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"连接智能助手服务失败 {agent_url}: {e}")
            st.markdown("服务可能正在启动中，请稍后重试。")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("未找到此对话ID的历史记录。")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # 配置选项
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        ""
        "基于 LangGraph、FastAPI 和 Streamlit 构建的 AI 智能助手系统"
        with st.popover(":material/settings: 设置", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("选择模型", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "选择助手",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("流式输出", value=True)

        @st.dialog("架构")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[在 Github 上查看完整大小](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "应用托管在 [Streamlit Cloud](https://share.streamlit.io/) 上，FastAPI 服务运行在 [Azure](https://learn.microsoft.com/en-us/azure/app-service/) 上"
            )

        if st.button(":material/schema: 架构", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: 隐私", use_container_width=True):
            st.write(
                "此应用中的提示、响应和反馈会被匿名记录并保存到 LangSmith，仅用于产品评估和改进。"
            )

        @st.dialog("分享/恢复对话")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}"
            st.markdown(f"**对话链接:**\n```text\n{chat_url}\n```")
            st.info("复制上面的链接以分享或恢复此对话")

        if st.button(":material/upload: 分享/恢复对话", use_container_width=True):
            share_chat_dialog()

        "[查看源代码](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "由 [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) 在 Oakland 制作"
        )

    # 绘制现有消息
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "你好！我是一个简单的聊天机器人。有什么我可以帮你的吗？"
            case "interrupt-agent":
                WELCOME = "你好！我是一个中断代理。告诉我你的生日，我可以预测你的性格！"
            case "research-assistant":
                WELCOME = "你好！我是一个具有网络搜索和计算器功能的 AI 研究助手。有什么我可以帮你的吗？"
            case _:
                WELCOME = "你好！我是一个 AI 助手。有什么我可以帮你的吗？"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() 需要一个异步迭代器
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # 如果用户提供了新输入，生成新消息
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                )
                messages.append(response)
                st.chat_message("ai").write(response.content)
            st.rerun()  # 清除过时的容器
        except AgentClientError as e:
            st.error(f"生成响应时出错: {e}")
            st.stop()

    # 如果生成了消息，显示反馈小部件
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()

async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """绘制一组聊天消息 - 重放现有消息或流式传输新消息。"""
    last_message_type = None
    st.session_state.last_message = None

    streaming_content = ""
    streaming_placeholder = None

    while msg := await anext(messages_agen, None):
        if isinstance(msg, str):
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"意外的消息类型: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            case "ai":
                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""工具调用: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("输入:")
                            status.write(tool_call["args"])

                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"意外的 ChatMessage 类型: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("输出:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("从助手接收到意外的 CustomData 消息")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            case _:
                st.error(f"意外的 ChatMessage 类型: {msg.type}")
                st.write(msg)
                st.stop()

async def handle_feedback() -> None:
    """绘制反馈小部件并记录用户的反馈。"""
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "行内用户反馈"},
            )
        except AgentClientError as e:
            st.error(f"记录反馈时出错: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("反馈已记录", icon=":material/reviews:")

if __name__ == "__main__":
    asyncio.run(main()) 