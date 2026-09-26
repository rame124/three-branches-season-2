"""Season 2 Three Branches agent: social villagers with role-based chores."""

from collections import deque

from sandbox.observation_types import ThreeBranchesAction, ThreeBranchesObservation
from sandbox.village import action, day, geometry, layout, me, people, props

WALK_SPEED = 0.75
CHORE_TICKS = 28
SOCIAL_TICKS = 10
SOCIAL_COOLDOWN = 70
STUCK_LIMIT = 5
WANDER_REST_TICKS = 18

MARKET_WORDS = ("stall", "shrine", "market")
FARM_WORDS = ("tending plot", "crop", "field", "farm", "garden", "planter", "plant")
LIGHT_WORDS = ("lamp", "lantern", "light", "torch")


def _cell_key(cell):
    return (cell["x"], cell["y"])


def _cell_dict(key):
    return {"x": key[0], "y": key[1]}


def _cell_centre(cell):
    return {"x": cell["x"] + 0.5, "y": cell["y"] + 0.5}


def _contains_any(prop, words):
    name = str(prop.get("type", "")).lower()
    return any(word in name for word in words)


class Agent:
    """Behavior-tree villager with a stable role and resumable chores."""

    def reset(self, seed: int, observation: ThreeBranchesObservation) -> None:
        self.rng = me.rng(observation, seed)
        self.player_id = me.player_id(observation)
        self.role = self._choose_role(self.player_id)

        self.current_chore = None
        self.completed_cycle = set()
        self.chore_ticks = 0

        self.route = []
        self.route_goal = None
        self.last_was_move = False
        self.stuck_ticks = 0
        self.yield_ticks = 0

        self.social_target = None
        self.social_ticks = 0
        self.social_cooldown_until = 0

        self.wander_target = None
        self.wander_rest = 0

        self.graph = {}
        self.outdoor_cells = []
        self._build_navigation(observation)
        self.role_props = self._role_props(observation)

    # ------------------------------------------------------------------
    # Behavior tree
    # ------------------------------------------------------------------

    def act(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Priority selector: home -> social -> resume chore -> begin chore -> wander."""

        for branch in (
            self._leave_home,
            self._react_socially,
            self._continue_chore,
            self._begin_ordinary_activity,
            self._wander,
        ):
            result = branch(observation)
            if result is not None:
                return result

        return action.stand(me.heading(observation), "none")

    # ------------------------------------------------------------------
    # Stable role assignment
    # ------------------------------------------------------------------

    def _choose_role(self, player_id):
        try:
            seat = int(str(player_id).split("_")[-1])
        except (TypeError, ValueError):
            seat = 1

        # cast_10: 1-4 market/shrine, 5-7 farms, 8-10 grounds/lights.
        if seat <= 4:
            return "market"
        if seat <= 7:
            return "farm"
        return "grounds"

    def _role_props(self, observation):
        all_props = list(props.all(observation))

        if self.role == "market":
            preferred = [p for p in all_props if _contains_any(p, MARKET_WORDS)]
        elif self.role == "farm":
            preferred = [p for p in all_props if _contains_any(p, FARM_WORDS)]
        else:
            preferred = [p for p in all_props if _contains_any(p, LIGHT_WORDS)]

        # Different villagers in the same job begin at different locations.
        preferred.sort(key=lambda p: str(p.get("id", "")))
        if preferred:
            offset = self._seat_number() % len(preferred)
            preferred = preferred[offset:] + preferred[:offset]

        return preferred

    def _seat_number(self):
        try:
            return int(str(self.player_id).split("_")[-1])
        except (TypeError, ValueError):
            return 1

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _build_navigation(self, observation):
        frame = layout.frame(observation)

        for x in range(frame["cells_x"]):
            for y in range(frame["cells_y"]):
                cell = {"x": x, "y": y}
                if not layout.walkable(observation, cell):
                    continue

                key = (x, y)
                self.graph[key] = []
                if layout.ground_at(observation, cell) != "interior":
                    self.outdoor_cells.append(key)

        for key in self.graph:
            start = _cell_dict(key)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                other = (key[0] + dx, key[1] + dy)
                if other not in self.graph:
                    continue
                if layout.can_step(observation, start, _cell_dict(other)):
                    self.graph[key].append(other)

    def _find_route(self, observation, goals, blocked=None):
        here_cell = layout.cell_at(observation, me.position(observation))
        if here_cell is None:
            return []

        start = _cell_key(here_cell)
        goals = set(goals)
        blocked = set(blocked or ())
        blocked.discard(start)
        if start in goals or start not in self.graph:
            return []

        queue = deque([start])
        previous = {start: None}
        found = None

        while queue:
            current = queue.popleft()
            if current in goals:
                found = current
                break
            for neighbour in self.graph.get(current, ()):
                if neighbour in blocked and neighbour not in goals:
                    continue
                if neighbour not in previous:
                    previous[neighbour] = current
                    queue.append(neighbour)

        if found is None:
            return []

        path = []
        current = found
        while current != start:
            path.append(current)
            current = previous[current]
        path.reverse()
        return path

    def _occupied_cells(self, observation):
        """Cells temporarily occupied by other characters we can perceive."""

        occupied = set()
        seen_ids = set()

        for person in people.seen(observation):
            seen_ids.add(person["id"])
            cell = layout.cell_at(observation, person["position"])
            if cell is not None:
                occupied.add(_cell_key(cell))

        for person in people.nearby(observation):
            if person["id"] in seen_ids:
                continue
            cell = layout.cell_at(observation, person["position"])
            if cell is not None:
                occupied.add(_cell_key(cell))

        return occupied

    def _nearby_blocking_npc(self, observation, distance=1.15):
        """Return a very close NPC, if one is currently blocking our movement."""

        here = me.position(observation)
        candidates = []
        seen_ids = set()

        for person in people.seen(observation):
            seen_ids.add(person["id"])
            if people.is_npc(person["id"]):
                candidates.append(person)

        for person in people.nearby(observation):
            if person["id"] in seen_ids or not people.is_npc(person["id"]):
                continue
            candidates.append(person)

        close = [
            p for p in candidates
            if geometry.distance(here, p["position"]) <= distance
        ]
        if not close:
            return None
        return min(close, key=lambda p: geometry.distance(here, p["position"]))

    def _move_to_cells(self, observation, goals, goal_id, expression="none"):
        here = me.position(observation)
        here_cell = layout.cell_at(observation, here)
        if here_cell is None:
            self.last_was_move = False
            return action.stand(me.heading(observation), expression)

        current = _cell_key(here_cell)

        if self.last_was_move:
            self.stuck_ticks = self.stuck_ticks + 1 if me.moved(observation) < 0.02 else 0

        if self.yield_ticks > 0:
            self.yield_ticks -= 1
            self.last_was_move = False
            return action.stand(me.heading(observation), expression)

        if self.stuck_ticks >= STUCK_LIMIT:
            self.route = []
            self.route_goal = None
            self.stuck_ticks = 0

            occupied = self._occupied_cells(observation)
            blocker = self._nearby_blocking_npc(observation)

            # If two villagers meet head-on, use player id as a stable
            # right-of-way rule. The higher-numbered villager yields/steps
            # aside, preventing both agents from making the same correction.
            should_yield = False
            if blocker is not None:
                try:
                    other_seat = int(str(blocker["id"]).split("_")[-1])
                    should_yield = self._seat_number() > other_seat
                except (TypeError, ValueError):
                    should_yield = True

            neighbours = [
                cell for cell in self.graph.get(current, [])
                if cell not in occupied
            ]

            # Prefer a side cell that is not simply the blocked next route
            # step. Stable ordering plus seat parity reduces mirrored choices.
            if neighbours:
                neighbours = sorted(neighbours)
                if len(neighbours) > 1:
                    index = self._seat_number() % len(neighbours)
                    neighbours = neighbours[index:] + neighbours[:index]
                target = _cell_centre(_cell_dict(neighbours[0]))
                self.last_was_move = True
                if should_yield:
                    self.yield_ticks = 2
                return action.walk(geometry.heading_to(here, target), WALK_SPEED * 0.8, expression)

            # No safe sidestep: briefly wait so the other villager can clear.
            if blocker is not None:
                self.yield_ticks = 3 + (self._seat_number() % 3)
                self.last_was_move = False
                return action.stand(me.heading(observation), expression)

        if self.route_goal != goal_id:
            self.route = []
            self.route_goal = goal_id

        if current in goals:
            self.last_was_move = False
            return None

        while self.route and self.route[0] == current:
            self.route.pop(0)

        if not self.route:
            self.route = self._find_route(
                observation, goals, blocked=self._occupied_cells(observation)
            )

        if not self.route:
            self.last_was_move = False
            return None

        target = _cell_centre(_cell_dict(self.route[0]))
        self.last_was_move = True
        return action.walk(geometry.heading_to(here, target), WALK_SPEED, expression)

    def _approach_cells(self, prop):
        x, y = prop["cell"]["x"], prop["cell"]["y"]
        return [
            cell
            for cell in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if cell in self.graph
        ]

    # ------------------------------------------------------------------
    # Behavior branches
    # ------------------------------------------------------------------

    def _leave_home(self, observation):
        here = me.position(observation)
        here_cell = layout.cell_at(observation, here)
        if here_cell is None or layout.ground_at(observation, here_cell) != "interior":
            return None

        home = me.home(observation)
        if home == "none":
            return None
        door = layout.doorway(observation, home)
        if door is None:
            return None

        self.last_was_move = True
        return action.walk(geometry.heading_to(here, door), WALK_SPEED, "none")

    def _react_socially(self, observation):
        tick = day.tick(observation)
        seen = list(people.seen(observation))
        if not seen:
            self.social_target = None
            self.social_ticks = 0
            return None

        # Visitor reactions outrank villager-to-villager socializing.
        visitors = [p for p in seen if people.is_visitor(p["id"])]
        waving_visitors = [
            p for p in visitors
            if p.get("expression", {}).get("type") == "wave"
        ]

        target = waving_visitors[0] if waving_visitors else None
        expression = "wave"

        # Also greet a nearby visitor occasionally even if they have not waved.
        if target is None and visitors and tick >= self.social_cooldown_until:
            target = min(
                visitors,
                key=lambda p: geometry.distance(me.position(observation), p["position"]),
            )
            if geometry.distance(me.position(observation), target["position"]) <= 3.5:
                expression = "wave"
            else:
                target = None

        # Villagers acknowledge each other, but only on a cooldown so the
        # entire cast does not get trapped in an endless greeting loop.
        if target is None and tick >= self.social_cooldown_until:
            npcs = [p for p in seen if people.is_npc(p["id"])]
            close_npcs = [
                p for p in npcs
                if geometry.distance(me.position(observation), p["position"]) <= 2.5
            ]
            if close_npcs and self.rng.random() < 0.12:
                target = self.rng.choice(close_npcs)
                expression = self.rng.choice(("nod", "wave", "laugh"))

        if target is None:
            return None

        if self.social_target != target["id"]:
            self.social_target = target["id"]
            self.social_ticks = 0

        self.social_ticks += 1
        here = me.position(observation)
        heading = geometry.heading_to(here, target["position"])
        self.last_was_move = False

        if self.social_ticks >= SOCIAL_TICKS:
            self.social_target = None
            self.social_ticks = 0
            self.social_cooldown_until = tick + SOCIAL_COOLDOWN

        return action.stand(heading, expression)

    def _continue_chore(self, observation):
        if self.current_chore is None:
            return None

        kind = self.current_chore["kind"]

        if kind == "prop":
            return self._perform_prop_chore(observation, self.current_chore["prop"])
        if kind == "sweep":
            return self._perform_sweep_chore(observation, self.current_chore["cell"])

        self.current_chore = None
        return None

    def _begin_ordinary_activity(self, observation):
        available = []
        for prop in self.role_props:
            if prop["id"] in self.completed_cycle:
                continue
            if self.role == "farm" and str(prop.get("type", "")).lower() == "tending plot":
                needs_tending = self._farm_plot_needs_tending(observation, prop)
                if needs_tending is False:
                    self.completed_cycle.add(prop["id"])
                    continue
            available.append(prop)

        if not available and self.role_props:
            # Start another work cycle instead of becoming permanently idle.
            # Farm plots are state-driven: after clearing the cycle, only
            # revisit plots that are unattended or currently out of sight.
            self.completed_cycle.clear()
            if self.role == "farm":
                for prop in self.role_props:
                    if str(prop.get("type", "")).lower() == "tending plot":
                        needs_tending = self._farm_plot_needs_tending(observation, prop)
                        if needs_tending is False:
                            continue
                    available.append(prop)
            else:
                available = list(self.role_props)

        if available:
            self.current_chore = {"kind": "prop", "prop": available[0]}
            self.chore_ticks = 0
            self.route = []
            self.route_goal = None
            return self._continue_chore(observation)

        # Groundskeepers can still visibly maintain the village even when
        # this map exposes no lamp/light props through props.all().
        if self.role == "grounds" and self.outdoor_cells:
            here_cell = layout.cell_at(observation, me.position(observation))
            here_key = _cell_key(here_cell) if here_cell else None
            choices = [c for c in self.outdoor_cells if c != here_key]
            if choices:
                self.current_chore = {"kind": "sweep", "cell": self.rng.choice(choices)}
                self.chore_ticks = 0
                self.route = []
                self.route_goal = None
                return self._continue_chore(observation)

        return None

    # ------------------------------------------------------------------
    # Chore execution
    # ------------------------------------------------------------------

    def _farm_plot_needs_tending(self, observation, prop):
        """Return True when a visible tending plot reports an unattended state.

        None means the plot is not visible yet, so callers should keep
        approaching it rather than assuming the chore is already complete.
        """

        if str(prop.get("type", "")).lower() != "tending plot":
            return True

        state = props.state(observation, prop["id"])
        if state is None:
            return None

        normalized = str(state).lower().replace("_", " ").replace("-", " ")
        needs_words = ("unattended", "untended", "needs tending", "needs attention")
        return any(word in normalized for word in needs_words)

    def _finish_chore(self, prop_id=None):
        if prop_id is not None:
            self.completed_cycle.add(prop_id)
        self.current_chore = None
        self.chore_ticks = 0
        self.route = []
        self.route_goal = None

    def _perform_prop_chore(self, observation, prop):
        goals = self._approach_cells(prop)
        if not goals:
            self._finish_chore(prop["id"])
            return None

        move = self._move_to_cells(observation, goals, ("prop", prop["id"]))
        if move is not None:
            return move

        here = me.position(observation)
        heading = geometry.heading_to(here, _cell_centre(prop["cell"]))

        # Farm workers only tend plots that actually need attention. Once a
        # tending plot is visible, skip it if another villager has already
        # taken care of it.
        if self.role == "farm" and str(prop.get("type", "")).lower() == "tending plot":
            needs_tending = self._farm_plot_needs_tending(observation, prop)
            if needs_tending is False:
                self._finish_chore(prop["id"])
                return None

        usable = props.usable(observation)
        self.chore_ticks += 1
        self.last_was_move = False

        if self.chore_ticks >= CHORE_TICKS:
            self._finish_chore(prop["id"])

        # Use the actual prop when possible. Otherwise show visible work.
        if usable is not None and usable.get("id") == prop["id"]:
            return action.stand(heading, "use")

        if self.role == "grounds":
            return action.stand(heading, "sweep")
        return action.stand(heading, "use")

    def _perform_sweep_chore(self, observation, cell):
        move = self._move_to_cells(observation, [cell], ("sweep", cell))
        if move is not None:
            return move

        self.chore_ticks += 1
        self.last_was_move = False
        if self.chore_ticks >= CHORE_TICKS:
            self._finish_chore()

        return action.stand(me.heading(observation), "sweep")

    # ------------------------------------------------------------------
    # Leisure / fallback
    # ------------------------------------------------------------------

    def _wander(self, observation):
        if not self.outdoor_cells:
            return None

        here_cell = layout.cell_at(observation, me.position(observation))
        if here_cell is None:
            return None
        current = _cell_key(here_cell)

        if self.wander_rest > 0:
            self.wander_rest -= 1
            self.last_was_move = False
            return action.stand(me.heading(observation), "none")

        if self.wander_target is None or self.wander_target == current:
            if self.wander_target == current:
                self.wander_rest = WANDER_REST_TICKS
            choices = [c for c in self.outdoor_cells if c != current]
            if not choices:
                return None
            self.wander_target = self.rng.choice(choices)
            self.route = []
            self.route_goal = None

        move = self._move_to_cells(
            observation,
            [self.wander_target],
            ("wander", self.wander_target),
        )
        if move is None:
            self.wander_target = None
        return move
