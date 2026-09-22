"""Cache policy for static files — a pure, framework-free module.

Vite stamps a content hash into build file names (`index-BI0ScPSE.js`). The
name changes when the content does, so those files are safe to cache forever.
A fixed name such as `index.html` must be revalidated every time, or a user
keeps seeing the old page after a deploy.

The StaticFiles mount in api.py consumes this decision.
"""

from __future__ import annotations

import re

#: Content-hashed files — one year, immutable (no revalidation at all).
IMMUTABLE = "public, max-age=31536000, immutable"

#: Fixed-name files — cached but revalidated on every request (ETag).
REVALIDATE = "public, max-age=0, must-revalidate"

#: Minimum bytes for gzip; below this the compression overhead is not worth it.
GZIP_MIN_SIZE = 1024

# Vite/rollup hash: `-<8 or more base64url chars>` before the extension.
_HASHED = re.compile(r"-[A-Za-z0-9_-]{8,}\.[A-Za-z0-9]+$")


def is_content_hashed(filename: str) -> bool:
    """Whether the file name carries a content hash."""
    if not filename:
        return False
    # Accept a path and look only at the last segment.
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    return bool(_HASHED.search(name))


def cache_control_for(filename: str) -> str:
    """Cache-Control value for a file name."""
    return IMMUTABLE if is_content_hashed(filename) else REVALIDATE
