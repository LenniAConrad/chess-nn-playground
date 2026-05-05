from __future__ import annotations

import re
from typing import Any


INFO_CP_RE = re.compile(r"\bscore cp (-?\d+)\b")
INFO_MATE_RE = re.compile(r"\bscore mate (-?\d+)\b")
INFO_NODES_RE = re.compile(r"\bnodes (\d+)\b")
INFO_PV_RE = re.compile(r"\bpv\s+(.+)$")


def parse_stockfish_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    engine = record.get("engine")
    if engine:
        metadata["stockfish_version"] = str(engine)
    analysis = record.get("analysis")
    if isinstance(analysis, list) and analysis:
        lines = [line for line in analysis if isinstance(line, str)]
        if lines:
            final = lines[-1]
            cp_match = INFO_CP_RE.search(final)
            mate_match = INFO_MATE_RE.search(final)
            nodes_match = INFO_NODES_RE.search(final)
            pv_match = INFO_PV_RE.search(final)
            if cp_match:
                metadata["pv1_cp"] = float(cp_match.group(1))
            if mate_match:
                metadata["pv1_mate"] = int(mate_match.group(1))
            if nodes_match:
                metadata["stockfish_nodes"] = int(nodes_match.group(1))
            if pv_match:
                moves = pv_match.group(1).split()
                if moves:
                    metadata["best_move"] = moves[0]
    return metadata
