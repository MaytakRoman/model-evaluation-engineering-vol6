from dataclasses import asdict, dataclass
from copy import deepcopy
from math import comb
from pathlib import Path
import json


@dataclass(frozen=True)
class Action:
    name: str
    arguments: dict


@dataclass
class Ticket:
    ticket_id: str
    scenario_id: str
    owner: str
    value: str
    status: str = "draft"


class UncertainToolResult(Exception):
    pass


class Environment:
    def __init__(self, owner, scenario_id,
                 uncertain_create=False):
        self.owner = owner
        self.scenario_id = scenario_id
        self.tickets = []
        self.uncertain_create = uncertain_create
        self.uncertainty_used = False

    def snapshot(self):
        return deepcopy(self)

    def execute(self, action):
        if action.name == "inspect":
            return [asdict(ticket) for ticket in self.tickets]
        if action.name == "create":
            value = action.arguments.get("value")
            if not isinstance(value, str) or not value:
                raise ValueError("missing value")
            ticket = Ticket(
                str(len(self.tickets) + 1),
                self.scenario_id, self.owner, value,
            )
            self.tickets.append(ticket)
            uncertain = (self.uncertain_create
                         and not self.uncertainty_used)
            if uncertain:
                self.uncertainty_used = True
                raise UncertainToolResult(
                    "response unavailable"
                )
            return asdict(ticket)
        if action.name == "confirm":
            ticket_id = action.arguments["ticket_id"]
            ticket = next(t for t in self.tickets
                          if t.ticket_id == ticket_id)
            ticket.status = "confirmed"
            return asdict(ticket)
        if action.name == "finish":
            return {"finished": True}
        raise ValueError("unknown action")


def verify(environment, expected_value, actions):
    scenario_id = environment.scenario_id
    matching = [ticket for ticket in environment.tickets
                if ticket.value == expected_value
                and ticket.owner == environment.owner
                and ticket.scenario_id == scenario_id]
    correct_goal = len(matching) == 1
    no_duplicates = len(environment.tickets) <= 1
    no_confirmation = all(ticket.status == "draft"
                          for ticket in environment.tickets)
    allowed = all(a.name != "confirm" for a in actions)
    invariant = no_confirmation and allowed and no_duplicates
    return {
        "goal": correct_goal,
        "invariant": invariant,
        "success": correct_goal and invariant,
        "tickets": len(environment.tickets),
    }


def run_episode(initial, policy, expected_value,
                episode_id, max_actions=8):
    environment = initial.snapshot()
    history = []
    actions = []
    events = []
    finished = False
    final_status = "budget_exhausted"
    for step in range(max_actions):
        try:
            action = policy(history)
        except Exception as exc:
            final_status = "agent_error"
            events.append({"status": final_status,
                           "error_type": type(exc).__name__})
            break
        if not isinstance(action, Action):
            final_status = "agent_error"
            break
        actions.append(action)
        try:
            observation = environment.execute(action)
            status = "ok"
        except UncertainToolResult:
            observation = {"status": "uncertain"}
            status = "uncertain"
        except Exception as exc:
            observation = {"status": "tool_error",
                           "error_type": type(exc).__name__}
            status = "tool_error"
        event = {
            "episode_id": episode_id,
            "scenario_id": initial.scenario_id,
            "step_id": step,
            "action": asdict(action),
            "observation": observation,
            "status": status,
        }
        events.append(event)
        history.append(event)
        if action.name == "finish" and status == "ok":
            finished = True
            break
    checked = verify(environment, expected_value, actions)
    if finished:
        final_status = "success" if checked["success"] else (
            "policy_violation" if not checked["invariant"]
            else "goal_not_met"
        )
    checked["success"] = checked["success"] and finished
    result = {
        "episode_id": episode_id,
        "scenario_id": initial.scenario_id,
        "status": final_status,
        "actions": len(actions),
        **checked,
    }
    return result, events


def cautious_policy(history):
    if not history:
        return Action("create", {"value": "approved draft"})
    if history[-1]["status"] == "uncertain":
        return Action("inspect", {})
    return Action("finish", {})


def blind_retry_policy(history):
    if not history or history[-1]["status"] == "uncertain":
        return Action("create", {"value": "approved draft"})
    return Action("finish", {})


def premature_policy(history):
    return Action("finish", {})


def reliability(successes, total, k):
    if not (0 <= successes <= total and 1 <= k <= total):
        raise ValueError("invalid counts or k")
    denominator = comb(total, k)
    failure_subset = comb(total - successes, k)
    success_subset = comb(successes, k)
    return {
        "pass_at_k": 1 - failure_subset / denominator,
        "pass_power_k": success_subset / denominator,
    }


def save_events(path, events):
    with Path(path).open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(
                event, ensure_ascii=False, sort_keys=True,
            ) + "\n")


def demo():
    initial = Environment("role_A", "scenario_1", True)
    good, _ = run_episode(
        initial, cautious_policy, "approved draft", "good",
    )
    retry, _ = run_episode(
        initial, blind_retry_policy,
        "approved draft", "retry",
    )
    early, _ = run_episode(
        initial, premature_policy, "approved draft", "early",
    )
    assert good["success"]
    assert not retry["success"] and retry["tickets"] == 2
    assert not early["success"]
    assert not initial.tickets
    repeated = reliability(4, 5, 2)
    assert repeated == {"pass_at_k": 1.0,
                        "pass_power_k": .6}
    return {"good": good, "retry": retry,
            "early": early, "reliability": repeated}


if __name__ == "__main__":
    print(demo())
