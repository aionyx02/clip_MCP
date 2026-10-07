"""Music from the web: only from sites whose terms are known, and only a song.

The sites are written here and nowhere else. Each says what its music may be
used for and whether a credit is owed: one licence for the whole site, or one
per track that the AI reads off the page and the user hears before anything is
downloaded. A link to anywhere else is refused, and so is a redirect that
leads anywhere else.
"""

import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse

# The biggest song taken: a long piece at a high bitrate fits, a film does not. Provisional.
MAX_BYTES = 50 * 1024 * 1024
TIMEOUT_SECONDS = 60
CHUNK_BYTES = 256 * 1024
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".flac")


@dataclass(frozen=True)
class Site:
    """A site music may be taken from.

    Attributes:
        name: What the user knows it as.
        hosts: Every host its pages and its files are served from.
        licence: The licence of all its music, or None when each track has its own.
        credit: Whether its licence asks for a credit.
        terms: What its licence allows, in a sentence for the user.
    """

    name: str
    hosts: Tuple[str, ...]
    licence: Optional[str]
    credit: bool
    terms: str


SITES = {
    "pixabay": Site(
        "Pixabay Music", ("pixabay.com", "cdn.pixabay.com"), "Pixabay Content License", False,
        "free for videos, commercial ones included, with no credit; the file may not be resold on its own. Some "
        "tracks are registered with Content ID: a claim is cleared with the licence certificate on the track's page",
    ),
    "mixkit": Site(
        "Mixkit", ("mixkit.co", "assets.mixkit.co"), "Mixkit Stock Music Free License", False,
        "free for videos online, social media and ads included, with no credit; not for TV or radio broadcast, "
        "CDs, DVDs or games",
    ),
    "incompetech": Site(
        "Incompetech (Kevin MacLeod)", ("incompetech.com",), "CC BY 4.0", True,
        "free for any video, monetized ones included, with a credit in the description",
    ),
    "fma": Site(
        "Free Music Archive", ("freemusicarchive.org", "files.freemusicarchive.org"), None, True,
        "each track has its own Creative Commons licence, shown on its page",
    ),
    "commons": Site(
        "Wikimedia Commons", ("commons.wikimedia.org", "upload.wikimedia.org"), None, True,
        "each file has its own free licence, shown on its page; all of them allow commercial use and changes",
    ),
}

# The licences a track on a per-track site may carry, as the AI names them.
LICENCES = ("CC0", "Public domain", "CC BY", "CC BY-SA", "CC BY-NC", "CC BY-ND", "CC BY-NC-SA", "CC BY-NC-ND")
# Those that keep a video from being commercial, or from cutting and looping the song.
LIMITING = ("NC", "ND")


def site_of(url: str) -> Optional[str]:
    """Which allowed site a link is on, or None when it is on none of them.

    Args:
        url: The link.

    Returns:
        The site's key in `SITES`.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    for key, site in SITES.items():
        if any(host == allowed or host.endswith("." + allowed) for allowed in site.hosts):
            return key
    return None


def licence_problem(site: str, licence: Optional[str], personal: bool) -> Optional[str]:
    """Why a track cannot be taken under this licence, or None when it can.

    Args:
        site: The site's key.
        licence: The licence the track's page shows, for a per-track site.
        personal: The user said the video is only for themselves.

    Returns:
        The reason, in a sentence for the AI.
    """
    if SITES[site].licence is not None:
        return None
    if licence not in LICENCES:
        return f"say the track's licence as shown on its page, one of: {', '.join(LICENCES)}"
    limiting = [part for part in LIMITING if part in licence.split("-")]
    if limiting and not personal:
        return (f"{licence} does not allow " + " or ".join({"NC": "commercial use", "ND": "changes such as "
                "cutting and looping"}[part] for part in limiting) + "; only for a video the user says is just for "
                "themselves, with `personal_use`")
    return None


def credit_line(site: str, title: str, artist: str, licence: str, page: str) -> Optional[str]:
    """The credit a track asks for, ready for a video's description, or None when it asks for none.

    Args:
        site: The site's key.
        title: The track's title.
        artist: Who made it.
        licence: Its licence.
        page: The page it came from.

    Returns:
        The line.
    """
    if site == "incompetech":
        return (f'"{title}" Kevin MacLeod (incompetech.com) Licensed under Creative Commons: By Attribution 4.0 '
                "https://creativecommons.org/licenses/by/4.0/")
    if not SITES[site].credit or licence in ("CC0", "Public domain"):
        return None
    return f'"{title}" by {artist or "unknown"} ({page}), licensed under {licence}'


class _StayOnSite(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to another allowed host: a download link may hand over to a CDN, nothing else."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        if site_of(new_url) is None:
            raise ValueError(f"the link leads to {urlparse(new_url).hostname}, which is not an allowed site")
        return super().redirect_request(request, response, code, message, headers, new_url)


def _file_name(url: str, disposition: Optional[str], title: str) -> str:
    """A safe file name for the song: what the server calls it, else the link's, else its title."""
    found = None
    if disposition:
        match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", disposition)
        found = unquote(match.group(1)) if match else None
    name = os.path.basename(found or unquote(urlparse(url).path)) or title
    stem, extension = os.path.splitext(name)
    if extension.lower() not in AUDIO_EXTENSIONS:
        stem, extension = name, ".mp3"
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title.strip() or stem).strip(" .") or "song"
    return stem[:120] + extension.lower()


def download(url: str, folder: str, title: str) -> str:
    """Download a song from an allowed site into a folder, never over another file.

    Args:
        url: The file's link.
        folder: Where to keep it.
        title: The track's title, for its file name.

    Returns:
        Where it was saved. Whether it is a song is for the caller to check.

    Raises:
        ValueError: If the link or a redirect leaves the allowed sites, or the
            file is bigger than `MAX_BYTES` or not sound.
        OSError: If the download fails.
    """
    from app.storage.housekeeping import unique_path

    if site_of(url) is None:
        raise ValueError("this link is not on one of the sites music may be taken from")
    opener = urllib.request.build_opener(_StayOnSite)
    request = urllib.request.Request(url, headers={"User-Agent": "clip-mcp (music for the user's own video)"})
    with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
        kind = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if kind.startswith(("text/", "image/", "video/")):
            raise ValueError(f"the link gives {kind}, not a song: link to the audio file itself, not its page")
        size = int(response.headers.get("Content-Length") or 0)
        if size > MAX_BYTES:
            raise ValueError(f"the file is {size / 1e6:.0f} MB, more than a song is allowed to be")
        os.makedirs(folder, exist_ok=True)
        target = unique_path(folder, _file_name(response.geturl(), response.headers.get("Content-Disposition"), title))
        part = target + ".part"
        written = 0
        try:
            with open(part, "wb") as kept:
                while chunk := response.read(CHUNK_BYTES):
                    written += len(chunk)
                    if written > MAX_BYTES:
                        raise ValueError("the file is more than a song is allowed to be")
                    kept.write(chunk)
            os.replace(part, target)
        finally:
            if os.path.exists(part):
                os.remove(part)
    return target
