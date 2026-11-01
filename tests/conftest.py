"""Register the vlab backend once per test session.

Registration is process-global and irreversible, so it happens here rather than
per-module; `vlab.register()` is idempotent for the same reason.
"""

import vlab

vlab.register()
