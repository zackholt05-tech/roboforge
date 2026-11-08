"""Container attestation of the held-out lock (D-067): prints, for each task, what
``eval_states.lock_status`` returns on the canonical container. It reads only the open
manifests and the freeze file, never a held-out state. Run once before the freeze commit
(expected: locked, the freeze file does not exist) and once after it (expected: unlocked,
attested by the container wrapper)."""
import json
import os
from datetime import UTC, datetime

from roboforge.provenance import evaluation_platform
from roboforge.sim import eval_states as es

out = {"utc": datetime.now(tz=UTC).isoformat(),
       "platform": evaluation_platform()["platform_id"],
       "host_git_sha": os.environ.get("ROBOFORGE_GIT_SHA"),
       "host_git_dirty": os.environ.get("ROBOFORGE_GIT_DIRTY"),
       "freeze_blob_from_host": os.environ.get("ROBOFORGE_FREEZE_BLOB") or None,
       "lock_status": {t: list(es.lock_status(t)) for t in ("lift", "can")}}
print(json.dumps(out, indent=1))
