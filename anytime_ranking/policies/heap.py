"""Serializable SELECT/TOP acquisition (Mohajer, Suh, Elmahdy, ICML 2017).

Repeated majority comparisons, balanced groups, tournament byes, and standard
binary max-heap operations. Completed TOP passes restart with fresh groups.
"""
from __future__ import annotations

import numpy as np


class TournamentHeap:
    def __init__(self, n, k, repetitions, state=None):
        if not 1 <= k <= n or repetitions < 1:
            raise ValueError("require 1 <= k <= n and positive repetitions")
        self.n, self.k, self.repetitions = n, k, repetitions
        if state is not None:
            self.state = state
        else:
            self.state = {'phase': 'restart', 'pass': 0, 'pending': None, 'last_order': []}

    def _selection(self, items, rng):
        self.state["selection"] = {
            "items": [int(i) for i in rng.permutation(items)], "next": [], "index": 0,
        }

    def _request(self, left, right, kind, candidate=-1):
        self.state["pending"] = {
            "left": left, "right": right, "kind": kind, "candidate": candidate,
            "count": 0, "wins": 0,
        }

    def _select_step(self):
        frame = self.state["selection"]
        while len(frame["items"]) > 1:
            items, index = frame["items"], frame["index"]
            if index >= len(items):
                frame.update(items=frame["next"], next=[], index=0)
            elif index + 1 == len(items):
                frame["next"].append(items[index])
                frame["index"] += 2
            else:
                self._request(items[index], items[index + 1], "tournament")
                return None
        return int(frame["items"][0])

    def _sift_step(self):
        frame, heap = self.state["sift"], self.state["heap"]
        while True:
            root = frame["root"]
            if frame["stage"] < 2:
                child = 2 * root + 1 + frame["stage"]
                if child < len(heap):
                    self._request(heap[frame["best"]], heap[child], "heap", child)
                    return False
                frame["stage"] += 1
            elif frame["best"] == root:
                return True
            else:
                best = frame["best"]
                heap[root], heap[best] = heap[best], heap[root]
                frame.update(root=best, best=best, stage=0)

    def next_pair(self, rng):
        # Phases run group tournaments, build heaps, extract winners, and repair heaps.
        # Return a pair when a new observation is needed and continue after observe().
        state = self.state
        while state["pending"] is None:
            phase = state["phase"]
            if phase == "restart":
                groups = [[int(i) for i in group] for group in np.array_split(rng.permutation(self.n), self.k)]
                home = [0] * self.n
                for group_index, group in enumerate(groups):
                    for item in group:
                        home[item] = group_index
                state.update(groups=groups, home=home, heap=[], output=[], group=0,
                             phase="group_select", sift=None)
                state["pass"] += 1
                self._selection(groups[0], rng)
            elif phase in {"group_select", "replace_select"}:
                winner = self._select_step()
                if winner is None:
                    break
                if phase == "group_select":
                    state["heap"].append(winner)
                    state["group"] += 1
                    if state["group"] < self.k:
                        self._selection(state["groups"][state["group"]], rng)
                    else:
                        state.update(phase="build_heap", build_index=len(state["heap"]) // 2 - 1)
                else:
                    state["heap"][0] = winner
                    state.update(phase="repair_heap", sift=None)
            elif phase in {"build_heap", "repair_heap"}:
                if phase == 'build_heap':
                    root = state['build_index']
                else:
                    root = 0
                if root < 0 or not state["heap"]:
                    state["phase"] = "extract"
                    continue
                if state["sift"] is None:
                    state["sift"] = {"root": root, "best": root, "stage": 0}
                if not self._sift_step():
                    break
                state["sift"] = None
                if phase == "build_heap":
                    state["build_index"] -= 1
                else:
                    state["phase"] = "extract"
            elif phase == "extract":
                winner = state["heap"][0]
                state["output"].append(winner)
                group = state["groups"][state["home"][winner]]
                group.remove(winner)
                if len(state["output"]) == self.k:
                    state.update(last_order=state["output"].copy(), phase="restart")
                elif group:
                    self._selection(group, rng)
                    state["phase"] = "replace_select"
                else:
                    last = state["heap"].pop()
                    if state["heap"]:
                        state["heap"][0] = last
                    state.update(phase="repair_heap", sift=None)
            else:
                raise RuntimeError(f"invalid TOP phase: {phase}")
        pending = state["pending"]
        return tuple(sorted((pending["left"], pending["right"])))

    def observe(self, pair, outcomes):
        pending = self.state["pending"]
        if pending is None or pair != tuple(sorted((pending["left"], pending["right"]))):
            raise ValueError("observation does not match the pending TOP comparison")
        pending["wins"] += sum(outcomes) if pending["left"] == pair[0] else len(outcomes) - sum(outcomes)
        pending["count"] += len(outcomes)
        if pending["count"] < self.repetitions:
            return
        if 2 * pending['wins'] >= pending['count']:
            winner = pending['left']
        else:
            winner = pending['right']
        if pending["kind"] == "tournament":
            self.state["selection"]["next"].append(winner)
            self.state["selection"]["index"] += 2
        else:
            candidate = pending["candidate"]
            if winner == self.state["heap"][candidate]:
                self.state["sift"]["best"] = candidate
            self.state["sift"]["stage"] += 1
        self.state["pending"] = None
