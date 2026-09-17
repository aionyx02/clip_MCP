import os
import subprocess
from typing import List
from app.models.timeline import Project

class FFmpegRenderer:
    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        self.ffmpeg_bin = ffmpeg_bin

    def build_command(self, project: Project, asset_map: dict[str, str], output_path: str, is_preview: bool = False) -> List[str]:
        cmd = [self.ffmpeg_bin, "-y"]
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

        filter_complex = f"{''.join(filter_input)}concat=n={input_count}:v=1:a=1[outv][outa]"
        cmd.extend(["-filter_complex", filter_complex, "-map", "[outv]", "-map", "[outa]"])

        if is_preview:
            cmd.extend([
                "-vf", "scale=-2:480",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-b:a", "96k"
            ])
        else:
            cmd.extend([
                "-r", f"{project.fps_npms}/{project.fps_den}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k"
            ])

        cmd.append(output_path)
        return cmd

    def execute(self, cmd: List[str], cancel_token: dict) -> subprocess.Popen:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False
        )