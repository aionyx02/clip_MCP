"""What every tool shares while the server runs: the database and its history, the job queue, the MCP server.

Also the workspace, the renderer, the skills folder, the tools named first and
who is calling. Opened once, at import, before any tool module is loaded. A tool
module reads them through this module (`runtime.repo`) rather than importing the
names, so there is one place for a test to swap one out. `mcp` is the exception:
every tool is registered with it and it is never swapped, so it is imported by
name.
"""

import os
from contextvars import ContextVar
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from fastmcp.server.transforms import ResourcesAsTools

from app.engine import machine, speed
from app.engine.builder import FFmpegRenderer
from app.engine.renderer import JobManager
from app.storage.repo import Repository
from app.workspace import workspace_dir

WORKSPACE_DIR = workspace_dir()
SKILLS_DIR = Path(__file__).parent / "skills"

# Opened before the server is described: what it says about this computer's speed comes from runs kept here.
repo = Repository(os.path.join(WORKSPACE_DIR, "clip_mcp.db"))
history = repo.keep_history(os.path.join(WORKSPACE_DIR, "history"))

# The tools an ordinary edit goes through, in the order they come up. Named in the opening
# instructions because a client that loads tools on demand has to guess which ones it will
# need before it has read anything; a test keeps every name here a real tool.
CORE_TOOLS = (
    "list_assets", "analyze_asset", "get_job", "build_semantic_timeline", "query_clips", "view_frames",
    "save_plan", "amend_plan", "create_project", "compile_plan", "preview_project", "render_project",
    "view_render", "inspect_join", "listen_again", "check_render", "apply_edits", "generate_subtitles",
)

mcp = FastMCP(
    "VideoEditingCore",
    instructions=(
        "Timeline-based video editing backed by FFmpeg. Before planning edits, "
        "read the clip-editing skill at skill://clip-editing/SKILL.md (clients "
        "without resource support can call read_resource with that URI). It "
        "explains how to turn editing requests into tool calls, how to read "
        "footage with analyze_asset, get_analysis, and view_frames, how to "
        "check an edit with preview_project before rendering, how to work "
        "through a folder of unedited footage when the user has no plan, "
        "what to ask when a request is incomplete, how music ducks under "
        "speech and how every render is normalized to one consistent "
        "loudness, how to caption an edit with generate_subtitles and "
        "burn_subtitles, how a render is checked first and refused until the "
        "user has heard what the check found, how one cut is delivered in "
        "several shapes and handed to other editing programs, "
        "and the current limits: no overlapping picture on a "
        "track, and no graphics beyond words: captions, title cards and text "
        "over the footage. Photos go in like videos and are shown for as long "
        "as their clip runs. "
        "Transitions end on the cut rather than straddling it, so adding one "
        "never changes how long the video runs. The person using this server is "
        "editing their own video, not writing code: reply in plain language "
        "with no tool names, IDs, or JSON, and read the plan back in one "
        "sentence for confirmation before the first render — unless they asked "
        "you to finish on your own, and then the skill says what still has to "
        "happen. Only files in the "
        "library can go into a project: a file the user names is added to it "
        "with import_asset or import_folder first, and you tell them it was "
        "added, since they see the library in the editor app and can take "
        "files out of it there; a file taken out is no longer usable until it "
        "is added again. Projects, assets, "
        "analyses, and jobs are saved on disk and survive server restarts. "
        f"The tools most edits go through, in order: {', '.join(CORE_TOOLS)}. "
        "Wherever a tool takes an asset ID, the file's exact name works too. "
        # Looked at once, at start: the AI has to know before it sends footage off how long that will take here.
        + machine.describe(machine.profile(), speed.estimate(repo, machine.profile(), {"accurate": 3600}, False))
    ),
)
# "resources" lists the supporting files alongside SKILL.md, so a client sees them
# without having to read the manifest first.
mcp.add_provider(SkillsDirectoryProvider(roots=SKILLS_DIR, supporting_files="resources"))
mcp.add_transform(ResourcesAsTools(mcp))

# The AI program making the current call, for what it says under its own name.
calling_client: ContextVar[str] = ContextVar("calling_client", default="AI 程式")

job_manager = JobManager(repo)
renderer = FFmpegRenderer()
