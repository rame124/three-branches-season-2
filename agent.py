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
FARM_WORDS = (
    "tending plot",
    "crop",
    "field",
    "farm",
    "garden",
    "planter",
    "plant",
)
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

    # ==============================================================
    # Behavior tree
    # ==============================================================

    def act(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Priority selector."""

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

    # ==============================================================
    # Roles
    # ==============================================================

    def _choose_role(self, player_id):
        try:
            seat = int(str(player_id).split("_")[-1])
        except (TypeError, ValueError):
            seat = 1

        # player_1 - player_4:
        # stalls and shrines
        if seat <= 4:
            return "market"

        # player_5 - player_7:
        # tending plots
        if seat <= 7:
            return "farm"

        # player_8 - player_10:
        # lights / sweeping
        return "grounds"

    def _role_props(self, observation):
        """Find every static prop relevant to this villager's role."""

        all_props = list(props.all(observation))

        if self.role == "market":
            preferred = [
                prop
                for prop in all_props
                if _contains_any(prop, MARKET_WORDS)
            ]

        elif self.role == "farm":
            preferred = [
                prop
                for prop in all_props
                if _contains_any(prop, FARM_WORDS)
            ]

        else:
            preferred = [
                prop
                for prop in all_props
                if _contains_any(prop, LIGHT_WORDS)
            ]

        preferred.sort(key=lambda prop: str(prop.get("id", "")))

        return preferred

    def _seat_number(self):
        try:
            return int(str(self.player_id).split("_")[-1])
        except (TypeError, ValueError):
            return 1

    # ==============================================================
    # Navigation graph
    # ==============================================================

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

            for dx, dy in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
            ):
                other = (
                    key[0] + dx,
                    key[1] + dy,
                )

                if other not in self.graph:
                    continue

                if layout.can_step(
                    observation,
                    start,
                    _cell_dict(other),
                ):
                    self.graph[key].append(other)

    def _find_route(
        self,
        observation,
        goals,
        blocked=None,
    ):
        """Breadth-first search toward any goal cell."""

        here_cell = layout.cell_at(
            observation,
            me.position(observation),
        )

        if here_cell is None:
            return []

        start = _cell_key(here_cell)

        goals = set(goals)
        blocked = set(blocked or ())

        blocked.discard(start)

        if start in goals:
            return []

        if start not in self.graph:
            return []

        queue = deque([start])

        previous = {
            start: None
        }

        found = None

        while queue:
            current = queue.popleft()

            if current in goals:
                found = current
                break

            for neighbour in self.graph.get(current, ()):
                if (
                    neighbour in blocked
                    and neighbour not in goals
                ):
                    continue

                if neighbour in previous:
                    continue

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

    # ==============================================================
    # Dynamic NPC avoidance
    # ==============================================================

    def _occupied_cells(self, observation):
        """Cells currently occupied by visible/heard characters."""

        occupied = set()
        seen_ids = set()

        for person in people.seen(observation):
            seen_ids.add(person["id"])

            cell = layout.cell_at(
                observation,
                person["position"],
            )

            if cell is not None:
                occupied.add(_cell_key(cell))

        for person in people.nearby(observation):
            if person["id"] in seen_ids:
                continue

            cell = layout.cell_at(
                observation,
                person["position"],
            )

            if cell is not None:
                occupied.add(_cell_key(cell))

        return occupied

    def _nearby_blocking_npc(
        self,
        observation,
        distance=1.15,
    ):
        """Return an NPC very close to us."""

        here = me.position(observation)

        candidates = []
        seen_ids = set()

        for person in people.seen(observation):
            seen_ids.add(person["id"])

            if people.is_npc(person["id"]):
                candidates.append(person)

        for person in people.nearby(observation):
            if person["id"] in seen_ids:
                continue

            if not people.is_npc(person["id"]):
                continue

            candidates.append(person)

        close = [
            person
            for person in candidates
            if geometry.distance(
                here,
                person["position"],
            ) <= distance
        ]

        if not close:
            return None

        return min(
            close,
            key=lambda person: geometry.distance(
                here,
                person["position"],
            ),
        )

    def _move_to_cells(
        self,
        observation,
        goals,
        goal_id,
        expression="none",
    ):
        here = me.position(observation)

        here_cell = layout.cell_at(
            observation,
            here,
        )

        if here_cell is None:
            self.last_was_move = False

            return action.stand(
                me.heading(observation),
                expression,
            )

        current = _cell_key(here_cell)

        # ----------------------------------------------------------
        # Detect failed movement
        # ----------------------------------------------------------

        if self.last_was_move:
            if me.moved(observation) < 0.02:
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = 0

        # Yield briefly if another NPC was given right-of-way.
        if self.yield_ticks > 0:
            self.yield_ticks -= 1
            self.last_was_move = False

            return action.stand(
                me.heading(observation),
                expression,
            )

        # ----------------------------------------------------------
        # Stuck recovery
        # ----------------------------------------------------------

        if self.stuck_ticks >= STUCK_LIMIT:
            self.route = []
            self.route_goal = None
            self.stuck_ticks = 0

            occupied = self._occupied_cells(observation)
            blocker = self._nearby_blocking_npc(observation)

            should_yield = False

            if blocker is not None:
                try:
                    other_seat = int(
                        str(blocker["id"]).split("_")[-1]
                    )

                    # Higher player number yields.
                    should_yield = (
                        self._seat_number() > other_seat
                    )

                except (TypeError, ValueError):
                    should_yield = True

            neighbours = [
                cell
                for cell in self.graph.get(current, [])
                if cell not in occupied
            ]

            if neighbours:
                neighbours = sorted(neighbours)

                # Different seats choose different sidesteps.
                if len(neighbours) > 1:
                    index = (
                        self._seat_number()
                        % len(neighbours)
                    )

                    neighbours = (
                        neighbours[index:]
                        + neighbours[:index]
                    )

                target = _cell_centre(
                    _cell_dict(neighbours[0])
                )

                self.last_was_move = True

                if should_yield:
                    self.yield_ticks = 2

                return action.walk(
                    geometry.heading_to(
                        here,
                        target,
                    ),
                    WALK_SPEED * 0.8,
                    expression,
                )

            # No safe sidestep.
            if blocker is not None:
                self.yield_ticks = (
                    3
                    + self._seat_number() % 3
                )

                self.last_was_move = False

                return action.stand(
                    me.heading(observation),
                    expression,
                )

        # ----------------------------------------------------------
        # Route management
        # ----------------------------------------------------------

        if self.route_goal != goal_id:
            self.route = []
            self.route_goal = goal_id

        # Already arrived.
        if current in goals:
            self.last_was_move = False
            return None

        while (
            self.route
            and self.route[0] == current
        ):
            self.route.pop(0)

        # First try avoiding NPCs.
        if not self.route:
            self.route = self._find_route(
                observation,
                goals,
                blocked=self._occupied_cells(
                    observation
                ),
            )

        # If NPC avoidance blocks everything,
        # find the normal static route.
        if not self.route:
            self.route = self._find_route(
                observation,
                goals,
            )

        # Still no route means genuinely unreachable.
        if not self.route:
            self.last_was_move = False
            self.route_goal = None

            return action.stand(
                me.heading(observation),
                expression,
            )

        target = _cell_centre(
            _cell_dict(self.route[0])
        )

        self.last_was_move = True

        return action.walk(
            geometry.heading_to(
                here,
                target,
            ),
            WALK_SPEED,
            expression,
        )

    # ==============================================================
    # Prop helpers
    # ==============================================================

    def _approach_cells(self, prop):
        """Walkable cells immediately surrounding a prop."""

        x = prop["cell"]["x"]
        y = prop["cell"]["y"]

        cardinal = [
            cell
            for cell in (
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            )
            if cell in self.graph
        ]

        if cardinal:
            return cardinal

        # Diagonal fallback.
        return [
            cell
            for cell in (
                (x + 1, y + 1),
                (x + 1, y - 1),
                (x - 1, y + 1),
                (x - 1, y - 1),
            )
            if cell in self.graph
        ]

    def _prop_position(self, prop):
        return _cell_centre(prop["cell"])

    def _prop_state_text(
        self,
        observation,
        prop,
    ):
        state = props.state(
            observation,
            prop["id"],
        )

        if state is None:
            return None

        return (
            str(state)
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

    def _visible_role_task_needed(
        self,
        observation,
        prop,
    ):
        """Return:
        True  -> visible and needs attention
        False -> visible and already correct
        None  -> cannot currently determine
        """

        state = self._prop_state_text(
            observation,
            prop,
        )

        if state is None:
            return None

        prop_type = str(
            prop.get("type", "")
        ).lower()

        # ----------------------------------------------------------
        # Farming
        # ----------------------------------------------------------

        if (
            self.role == "farm"
            and prop_type == "tending plot"
        ):
            if any(
                word in state
                for word in (
                    "overgrown",
                    "unattended",
                    "untended",
                    "needs tending",
                    "needs attention",
                )
            ):
                return True

            # Once the plot changes away from the bad state,
            # consider it done.
            return False

        # ----------------------------------------------------------
        # Market / shrine
        # ----------------------------------------------------------

        if self.role == "market":

            if (
                "stall" in prop_type
                or "market" in prop_type
            ):
                # IMPORTANT:
                # Leave an open stall alone.
                if (
                    "open" in state
                    and "closed" not in state
                ):
                    return False

                if "closed" in state:
                    return True

            if "shrine" in prop_type:

                if (
                    any(
                        word in state
                        for word in (
                            "tended",
                            "attended",
                        )
                    )
                    and not any(
                        word in state
                        for word in (
                            "untended",
                            "unattended",
                        )
                    )
                ):
                    return False

                if any(
                    word in state
                    for word in (
                        "untended",
                        "unattended",
                        "overgrown",
                        "unlit",
                        "needs tending",
                        "needs attention",
                    )
                ):
                    return True

        # ----------------------------------------------------------
        # Grounds / lights
        # ----------------------------------------------------------

        if self.role == "grounds":

            if any(
                word in prop_type
                for word in LIGHT_WORDS
            ):
                if (
                    any(
                        word in state
                        for word in (
                            "lit",
                            "on",
                        )
                    )
                    and not any(
                        word in state
                        for word in (
                            "unlit",
                            "off",
                        )
                    )
                ):
                    return False

                if any(
                    word in state
                    for word in (
                        "unlit",
                        "off",
                        "dark",
                    )
                ):
                    return True

        return None

    # ==============================================================
    # Chore selection
    # ==============================================================

    def _select_role_prop(
        self,
        observation,
        visible_only=False,
    ):
        """Pick the nearest useful role prop.

        Visible props that definitely need work are preferred
        over unknown distant props.
        """

        here = me.position(observation)

        candidates = []

        for prop in self.role_props:

            if prop["id"] in self.completed_cycle:
                continue

            needed = self._visible_role_task_needed(
                observation,
                prop,
            )

            # If we can see the prop and it is already fine,
            # mark it complete for this patrol.
            if needed is False:
                self.completed_cycle.add(
                    prop["id"]
                )
                continue

            if (
                visible_only
                and needed is not True
            ):
                continue

            distance = geometry.distance(
                here,
                self._prop_position(prop),
            )

            # Visible confirmed work is priority 0.
            # Unknown work is priority 1.
            priority = (
                0
                if needed is True
                else 1
            )

            candidates.append(
                (
                    priority,
                    distance,
                    str(prop.get("id", "")),
                    prop,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[:3]
        )

        return candidates[0][3]

    def _maybe_preempt_for_nearby_work(
        self,
        observation,
    ):
        """Switch to visible nearby work instead of walking past it."""

        if self.current_chore is None:
            return

        if (
            self.current_chore.get("kind")
            != "prop"
        ):
            return

        nearby = self._select_role_prop(
            observation,
            visible_only=True,
        )

        if nearby is None:
            return

        current = self.current_chore["prop"]

        if nearby["id"] == current["id"]:
            return

        here = me.position(observation)

        nearby_distance = geometry.distance(
            here,
            self._prop_position(nearby),
        )

        current_distance = geometry.distance(
            here,
            self._prop_position(current),
        )

        # If visible work is meaningfully closer,
        # interrupt the speculative destination.
        if (
            nearby_distance + 0.5
            < current_distance
        ):
            self.current_chore = {
                "kind": "prop",
                "prop": nearby,
            }

            self.chore_ticks = 0
            self.route = []
            self.route_goal = None

    # ==============================================================
    # Behavior branches
    # ==============================================================

    def _leave_home(self, observation):
        here = me.position(observation)

        here_cell = layout.cell_at(
            observation,
            here,
        )

        if here_cell is None:
            return None

        if (
            layout.ground_at(
                observation,
                here_cell,
            )
            != "interior"
        ):
            return None

        home = me.home(observation)

        if home == "none":
            return None

        door = layout.doorway(
            observation,
            home,
        )

        if door is None:
            return None

        self.last_was_move = True

        return action.walk(
            geometry.heading_to(
                here,
                door,
            ),
            WALK_SPEED,
            "none",
        )

    def _react_socially(
        self,
        observation,
    ):
        tick = day.tick(observation)

        seen = list(
            people.seen(observation)
        )

        if not seen:
            self.social_target = None
            self.social_ticks = 0
            return None

        # ----------------------------------------------------------
        # Visitors have priority
        # ----------------------------------------------------------

        visitors = [
            person
            for person in seen
            if people.is_visitor(
                person["id"]
            )
        ]

        waving_visitors = [
            person
            for person in visitors
            if (
                person
                .get("expression", {})
                .get("type")
                == "wave"
            )
        ]

        target = (
            waving_visitors[0]
            if waving_visitors
            else None
        )

        expression = "wave"

        # Occasionally greet a visitor who is nearby
        # even if they did not wave first.
        if (
            target is None
            and visitors
            and tick
            >= self.social_cooldown_until
        ):
            target = min(
                visitors,
                key=lambda person:
                    geometry.distance(
                        me.position(observation),
                        person["position"],
                    ),
            )

            if (
                geometry.distance(
                    me.position(observation),
                    target["position"],
                )
                > 3.5
            ):
                target = None

        # ----------------------------------------------------------
        # NPC-to-NPC social activity
        # ----------------------------------------------------------

        if (
            target is None
            and tick
            >= self.social_cooldown_until
        ):
            npcs = [
                person
                for person in seen
                if people.is_npc(
                    person["id"]
                )
            ]

            close_npcs = [
                person
                for person in npcs
                if geometry.distance(
                    me.position(observation),
                    person["position"],
                )
                <= 2.5
            ]

            if (
                close_npcs
                and self.rng.random() < 0.12
            ):
                target = self.rng.choice(
                    close_npcs
                )

                expression = self.rng.choice(
                    (
                        "nod",
                        "wave",
                        "laugh",
                    )
                )

        if target is None:
            return None

        if (
            self.social_target
            != target["id"]
        ):
            self.social_target = (
                target["id"]
            )

            self.social_ticks = 0

        self.social_ticks += 1

        heading = geometry.heading_to(
            me.position(observation),
            target["position"],
        )

        self.last_was_move = False

        if (
            self.social_ticks
            >= SOCIAL_TICKS
        ):
            self.social_target = None
            self.social_ticks = 0

            self.social_cooldown_until = (
                tick + SOCIAL_COOLDOWN
            )

        return action.stand(
            heading,
            expression,
        )

    def _continue_chore(
        self,
        observation,
    ):
        if self.current_chore is None:
            return None

        self._maybe_preempt_for_nearby_work(
            observation
        )

        kind = self.current_chore["kind"]

        if kind == "prop":
            return self._perform_prop_chore(
                observation,
                self.current_chore["prop"],
            )

        if kind == "sweep":
            return self._perform_sweep_chore(
                observation,
                self.current_chore["cell"],
            )

        self.current_chore = None

        return None

    def _begin_ordinary_activity(
        self,
        observation,
    ):
        target = self._select_role_prop(
            observation
        )

        # If the entire previous patrol is finished,
        # start another inspection cycle.
        if (
            target is None
            and self.role_props
        ):
            self.completed_cycle.clear()

            target = self._select_role_prop(
                observation
            )

        if target is not None:
            self.current_chore = {
                "kind": "prop",
                "prop": target,
            }

            self.chore_ticks = 0
            self.route = []
            self.route_goal = None

            return self._continue_chore(
                observation
            )

        # Groundskeepers can sweep even if no
        # explicit light props exist.
        if (
            self.role == "grounds"
            and self.outdoor_cells
        ):
            here_cell = layout.cell_at(
                observation,
                me.position(observation),
            )

            here_key = (
                _cell_key(here_cell)
                if here_cell
                else None
            )

            choices = [
                cell
                for cell
                in self.outdoor_cells
                if cell != here_key
            ]

            if choices:
                self.current_chore = {
                    "kind": "sweep",
                    "cell": self.rng.choice(
                        choices
                    ),
                }

                self.chore_ticks = 0
                self.route = []
                self.route_goal = None

                return self._continue_chore(
                    observation
                )

        return None

    # ==============================================================
    # Chore execution
    # ==============================================================

    def _finish_chore(
        self,
        prop_id=None,
    ):
        if prop_id is not None:
            self.completed_cycle.add(
                prop_id
            )

        self.current_chore = None
        self.chore_ticks = 0
        self.route = []
        self.route_goal = None

    def _perform_prop_chore(
        self,
        observation,
        prop,
    ):
        goals = self._approach_cells(prop)

        if not goals:
            self.route = []
            self.route_goal = None

            return action.stand(
                me.heading(observation),
                "none",
            )

        move = self._move_to_cells(
            observation,
            goals,
            (
                "prop",
                prop["id"],
            ),
        )

        if move is not None:
            return move

        # ----------------------------------------------------------
        # We reached an approach cell.
        # ----------------------------------------------------------

        here = me.position(observation)

        prop_position = self._prop_position(
            prop
        )

        heading = geometry.heading_to(
            here,
            prop_position,
        )

        prop_type = str(
            prop.get("type", "")
        ).lower()

        is_tending_plot = (
            self.role == "farm"
            and prop_type == "tending plot"
        )

        # ----------------------------------------------------------
        # Re-check state before touching anything
        # ----------------------------------------------------------

        state_needed = (
            self._visible_role_task_needed(
                observation,
                prop,
            )
        )

        # Visible and already correct.
        if state_needed is False:
            self._finish_chore(
                prop["id"]
            )

            return None

        usable = props.usable(
            observation
        )

        # ----------------------------------------------------------
        # Creep closer until the exact prop is usable.
        # ----------------------------------------------------------

        if (
            usable is None
            or usable.get("id")
            != prop["id"]
        ):
            self.last_was_move = True

            return action.walk(
                heading,
                0.20,
                "none",
            )

        self.last_was_move = False

        # ----------------------------------------------------------
        # Tending plots
        # ----------------------------------------------------------

        if is_tending_plot:
            # Never finish by timer.
            #
            # Keep issuing use until the visible state
            # changes away from "overgrown".
            return action.stand(
                heading,
                "use",
            )

        # ----------------------------------------------------------
        # Toggle-style props
        # ----------------------------------------------------------

        is_toggle_style = (
            (
                self.role == "market"
                and (
                    "stall" in prop_type
                    or "market" in prop_type
                )
            )
            or (
                self.role == "grounds"
                and any(
                    word in prop_type
                    for word in LIGHT_WORDS
                )
            )
        )

        if is_toggle_style:
            # One use only.
            #
            # Immediately stop the chore after using it.
            # On the next patrol the prop state will be checked
            # again before anyone touches it.
            self._finish_chore(
                prop["id"]
            )

            return action.stand(
                heading,
                "use",
            )

        # ----------------------------------------------------------
        # Shrines / generic work
        # ----------------------------------------------------------

        self.chore_ticks += 1

        if (
            self.chore_ticks
            >= CHORE_TICKS
        ):
            self._finish_chore(
                prop["id"]
            )

        return action.stand(
            heading,
            "use",
        )

    def _perform_sweep_chore(
        self,
        observation,
        cell,
    ):
        move = self._move_to_cells(
            observation,
            [cell],
            (
                "sweep",
                cell,
            ),
        )

        if move is not None:
            return move

        self.chore_ticks += 1
        self.last_was_move = False

        if (
            self.chore_ticks
            >= CHORE_TICKS
        ):
            self._finish_chore()

        return action.stand(
            me.heading(observation),
            "sweep",
        )

    # ==============================================================
    # Wandering
    # ==============================================================

    def _wander(
        self,
        observation,
    ):
        if not self.outdoor_cells:
            return None

        here_cell = layout.cell_at(
            observation,
            me.position(observation),
        )

        if here_cell is None:
            return None

        current = _cell_key(
            here_cell
        )

        if self.wander_rest > 0:
            self.wander_rest -= 1
            self.last_was_move = False

            return action.stand(
                me.heading(observation),
                "none",
            )

        if (
            self.wander_target is None
            or self.wander_target == current
        ):
            if (
                self.wander_target
                == current
            ):
                self.wander_rest = (
                    WANDER_REST_TICKS
                )

            choices = [
                cell
                for cell
                in self.outdoor_cells
                if cell != current
            ]

            if not choices:
                return None

            self.wander_target = (
                self.rng.choice(
                    choices
                )
            )

            self.route = []
            self.route_goal = None

        move = self._move_to_cells(
            observation,
            [self.wander_target],
            (
                "wander",
                self.wander_target,
            ),
        )

        if move is None:
            self.wander_target = None

        return move