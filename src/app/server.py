import threading
import time

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from app.models.plan import short_clip_ids
from app import runtime
# By name, unlike everything else from runtime: every tool is registered with it, and it is never swapped out.
from app.runtime import mcp
# Importing a tool module is what registers its tools, so every one is imported here.
from app.tools import (  # noqa: F401
    catalog, delivery, editing, examining, feedback, job_queue, looking, planning, rendering, shared, soundtrack,
    subtitles, timelines, upkeep, versioning,
)

def _shortened(value: object) -> object:
    """Take the timeline off every clip ID in a result, however deep it sits.

    Args:
        value: A result, or any part of one.

    Returns:
        The same shape, with every string's clip IDs short.
    """
    if isinstance(value, str):
        return short_clip_ids(value)
    if isinstance(value, list):
        return [_shortened(item) for item in value]
    if isinstance(value, dict):
        return {key: _shortened(item) for key, item in value.items()}
    return value

class ShortClipIds(Middleware):
    """Show the AI every clip ID one way: `u0034`, without the timeline in front.

    A clip's whole ID is its timeline's and its own, and the AI used to be shown the
    whole one in some places and the short one in others, then had to write the whole
    one into a plan. Everything said back passes through here, errors included, and
    every tool that takes a clip knows which timeline it means.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> ToolResult:
        """Run the tool, then shorten every clip ID in what it said or raised.

        Args:
            context: The call.
            call_next: The rest of the chain, ending in the tool.

        Returns:
            The tool's result with short clip IDs.

        Raises:
            ToolError: The tool's own error, its clip IDs short.
        """
        try:
            result = await call_next(context)
        except Exception as exc:
            said = str(exc)
            if short_clip_ids(said) != said:
                raise ToolError(short_clip_ids(said)) from exc
            raise
        if not isinstance(result, ToolResult):
            return result
        return ToolResult(
            content=[
                TextContent(type="text", text=short_clip_ids(block.text)) if isinstance(block, TextContent) else block
                for block in result.content
            ],
            structured_content=_shortened(result.structured_content),
            meta=result.meta,
        )

mcp.add_middleware(ShortClipIds())

def _with_asset_ids(value: object) -> object:
    """Put an ID in place of every file name given as an asset, however deep it sits.

    Args:
        value: A tool's arguments, or any part of them.

    Returns:
        The same shape, with each `asset_id` and `asset_ids` entry an ID
        where it named exactly one file.

    Raises:
        ValueError: If a name is shared by more than one file.
    """
    if isinstance(value, list):
        return [_with_asset_ids(item) for item in value]
    if not isinstance(value, dict):
        return value
    named = {}
    for key, item in value.items():
        if key == "asset_id":
            named[key] = shared.asset_id_for(item)
        elif key == "asset_ids" and isinstance(item, list):
            named[key] = [shared.asset_id_for(name) for name in item]
        else:
            named[key] = _with_asset_ids(item)
    return named

class AssetNames(Middleware):
    """Take a file's exact name wherever a tool takes an asset ID.

    The AI reads footage by file name — in a search's lines, in what the user
    says — and used to look each one up to get the ID a tool wanted. A name
    that is one file's becomes its ID; anything else goes on as it came.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> ToolResult:
        """Swap file names for IDs in the arguments, then run the tool.

        Args:
            context: The call.
            call_next: The rest of the chain, ending in the tool.

        Returns:
            What the tool returned.

        Raises:
            ToolError: If a name is shared by more than one file.
        """
        arguments = context.message.arguments
        if arguments:
            try:
                named = _with_asset_ids(arguments)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            context = context.copy(message=context.message.model_copy(update={"arguments": named}))
        return await call_next(context)

mcp.add_middleware(AssetNames())

# What each tool that changes things is doing, for a version saved without the user's words.
DOING = {
    "apply_edits": "修改時間軸", "compile_plan": "依計畫剪接", "save_plan": "存計畫", "amend_plan": "修改計畫",
    "revert_plan": "計畫回到舊版", "copy_plan": "複製計畫", "generate_subtitles": "上字幕",
    "create_project": "建立專案", "import_asset": "加入素材", "import_folder": "加入資料夾", "edit_asset": "寫素材備註",
    "restore_version": "回到舊版本", "branch_project": "開分支", "mark_version": "標記版本",
    "organize_library": "整理素材庫", "move_files": "搬動檔案", "set_asset_library": "移動分類",
    "find_music": "找歌", "compare_versions": "比較版本", "merge_version_parts": "合成兩版", "add_music_from_url": "從網路加入音樂", "resolve_comment": "回覆留言",
}
# What the AI programs call themselves when they connect, as a person would name them.
CLIENT_NAMES = {
    "claude-code": "Claude Code", "claude-ai": "Claude Desktop", "codex-mcp-client": "Codex",
    "gemini-cli-mcp-client": "Gemini CLI", "opencode": "opencode", "cherry-studio": "Cherry Studio",
    "lm-studio": "LM Studio",
}

def _client_name(context: MiddlewareContext) -> str:
    """The AI program on the other end of a call, by the name it gave when it connected."""
    try:
        params = context.fastmcp_context.session.client_params
        name = params.client_info.name if params is not None and params.client_info is not None else ""
    except (AttributeError, RuntimeError):
        name = ""
    return CLIENT_NAMES.get(name.lower(), name) or "AI 程式"

class HistorySteps(Middleware):
    """Keep what one tool call changed as one version, said in words, under the AI's name."""

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> ToolResult:
        """Run the tool inside one step of the history.

        Args:
            context: The call; its `note` argument, when there is one, is the
                version's description.
            call_next: The rest of the chain, ending in the tool.

        Returns:
            What the tool returned.
        """
        arguments = context.message.arguments or {}
        note = arguments.get("note") if isinstance(arguments.get("note"), str) else ""
        name = context.message.name
        client = _client_name(context)
        token = runtime.calling_client.set(client)
        try:
            with runtime.history.step(f"AI（{client}）", note=note, doing=DOING.get(name, name)):
                return await call_next(context)
        finally:
            runtime.calling_client.reset(token)

mcp.add_middleware(HistorySteps())

# Tools that already are about the comments, so are not reminded of them.
ABOUT_COMMENTS = {"get_comments", "resolve_comment"}

class OpenComments(Middleware):
    """Remind the AI, on anything it does to a video, of what the user said about it and is still waiting.

    The editor cannot call the AI: a comment written there waits until the
    AI next looks. So every tool that names a project says how many are
    waiting, whether or not the user remembered to mention them.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> ToolResult:
        """Run the tool, then add the count of open comments on the project it was about.

        Args:
            context: The call.
            call_next: The rest of the chain, ending in the tool.

        Returns:
            The tool's result, with `open_comments` when there are any.
        """
        result = await call_next(context)
        project_id = (context.message.arguments or {}).get("project_id")
        if (context.message.name in ABOUT_COMMENTS or not isinstance(project_id, str)
                or not isinstance(result, ToolResult)):
            return result
        try:
            waiting = open_comments(project_id)
        except Exception:
            # A reminder must never cost the tool its answer: whatever went wrong reading the
            # comments, the tool's own result still goes back as it was.
            return result
        if not waiting:
            return result
        said = (f"the user left {waiting} comment(s) on this video in the editor that are not dealt with yet: "
                "read them with `get_comments`, and close each with `resolve_comment` once done")
        structured = result.structured_content
        if isinstance(structured, dict):
            structured = {**structured, "open_comments": {"count": waiting, "say": said}}
        return ToolResult(content=[*result.content, TextContent(type="text", text=said)],
                          structured_content=structured, meta=result.meta)

mcp.add_middleware(OpenComments())

def open_comments(project_id: str) -> int:
    """How many things the user said about a project are still waiting.

    A question the AI asked waits on the user until they answer it, so it is
    not counted before then.

    Args:
        project_id: The video.

    Returns:
        The number of open comments.
    """
    return sum(comment.waiting for comment in runtime.repo.comments(project_id))

# How often a running server moves the job queue along. Provisional.
TEND_JOBS_SECONDS = 15.0

def tend_jobs() -> None:
    """Keep the job queue moving while the server runs.

    A worker starts the next queued job when it finishes, but one killed
    halfway never does, and what was queued behind it waited for some later
    call to look (#88). Run at start and then every `TEND_JOBS_SECONDS`: a
    job whose worker is gone is failed once it has been quiet long enough,
    and the queue behind it starts.
    """
    while True:
        try:
            runtime.job_manager.launch_ready()
        except Exception:
            # A failed look is tried again next time; it is never the end of the server.
            pass
        time.sleep(TEND_JOBS_SECONDS)

def main():
    """Run the MCP server over the stdio transport."""
    threading.Thread(target=catalog.tend_library, name="tend-library", daemon=True).start()
    threading.Thread(target=tend_jobs, name="tend-jobs", daemon=True).start()
    # The banner goes to stderr, which every client keeps as its server log; a box of
    # ASCII art in it on every start is noise, and on a console that is not UTF-8 it
    # arrives mangled.
    mcp.run(show_banner=False)

if __name__ == "__main__":
    main()
