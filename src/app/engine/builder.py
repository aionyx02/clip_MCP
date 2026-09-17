import os
import subprocess
from typing import List
from app.models.timeline import Project

class FFmpegRenderer:
    """Translates a project timeline into an FFmpeg command and runs it."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        """Initialize the renderer.

        Args:
            ffmpeg_bin: Path to, or name of, the FFmpeg executable.
        """
        self.ffmpeg_bin = ffmpeg_bin

    def build_command(self, project: Project, asset_map: dict[str, str], output_path: str, is_preview: bool = False) -> List[str]:
        """Build the FFmpeg arguments that render a project to a file.

        Each clip is trimmed to its source range, and all clips are
        concatenated in track order into one video and one audio stream, so
        every source must contain both. Previews are scaled to 480p and
        encoded for speed; final renders use the project frame rate and
        higher-quality encoder settings.

        Args:
            project: Project whose timeline is rendered.
            asset_map: Mapping of asset IDs to source media file paths.
            output_path: Destination path of the rendered file.
            is_preview: Whether to render a fast, low-resolution preview.

        Returns:
            The command as an argument list suitable for `subprocess.Popen`.

        Raises:
            KeyError: If a clip references an asset ID missing from `asset_map`.
            ValueError: If the project contains no clips.
        """
        # Only log errors and suppress progress stats, so the unread stderr pipe cannot fill up and block FFmpeg.
        cmd = [self.ffmpeg_bin, "-y", "-loglevel", "error", "-nostats"]
        filter_input = []
        input_count = 0

        for track in project.tracks:
            for clip in track.clips:
                src_path = asset_map[clip.asset_id]

                cmd.extend([
                    "-ss", str(clip.source_range.start),
                    "-to", str(clip.source_range.end),
                    "-i", src_path
                ])

                filter_input.append(f"[{input_count}:v][{input_count}:a]")
                input_count += 1

        if input_count == 0:
            raise ValueError("No valid input clips found")

        if is_preview:
            # -vf cannot be combined with -filter_complex on the same stream, so scale inside the graph.
            filter_complex = f"{''.join(filter_input)}concat=n={input_count}:v=1:a=1[catv][outa];[catv]scale=-2:480[outv]"
        else:
            filter_complex = f"{''.join(filter_input)}concat=n={input_count}:v=1:a=1[outv][outa]"
        cmd.extend(["-filter_complex", filter_complex, "-map", "[outv]", "-map", "[outa]"])

        if is_preview:
            cmd.extend([
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-b:a", "96k"
            ])
        else:
            cmd.extend([
                "-r", f"{project.fps_num}/{project.fps_den}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k"
            ])

        cmd.append(output_path)
        return cmd

    def execute(self, cmd: List[str], cancel_token: dict) -> subprocess.Popen:
        """Start an FFmpeg process without waiting for it to finish.

        Standard input is detached so the child process cannot read from the
        MCP stdio transport.

        Args:
            cmd: Command produced by `build_command`.
            cancel_token: Reserved for cooperative cancellation; currently unused.

        Returns:
            The running process, with stdout and stderr captured through pipes.
        """
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False
        )
