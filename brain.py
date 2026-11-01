"""Astra 6: the language model that turns operator instructions into robot action plans.

Two interchangeable backends:
  - Claude, through the Anthropic SDK (key from ANTHROPIC_API_KEY or an `ant auth login` profile)
  - any OpenAI-compatible chat endpoint (key from OPENAI_API_KEY; base URL configurable)
"""
import base64
import json
import os
import re

import anthropic
import openai

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_OPENAI_MODEL = os.environ.get("ASTRA_OPENAI_MODEL", "gpt-4.1")

SYSTEM = """You are Astra 6, the planning model that controls a Franka Panda robot arm with a parallel two-finger gripper in a MuJoCo physics simulation of a tabletop cell. The operator gives instructions in natural language; you turn each one into a short plan of robot actions.

Frame: meters, robot base at the origin, +x away from the robot, +y is the robot's left. "left" and "right" always mean the robot's left (+y) and right (-y), which is also how the operator sees the cell from behind the robot.

What you know about the cell comes from the robot's camera, not from the simulator:
- The object list is the vision system's estimate: colour detection plus depth, fitted to known object shapes. Positions are accurate to a few millimetres. An object can be hidden (for example behind the arm); then its last seen position is given.
- You also get the camera image itself, with a labelled box on each detected object and a tag on each bin. The camera is at x 1.20, y 0.00, height 0.75, in front of the table looking back at the robot, so the robot's left (+y) appears on the RIGHT side of the image.
- Use the image for what things look like (colour, which bin is which). For distances and "nearest / farthest / closest to X", compute from the listed coordinates (and the camera position above), not from how the picture looks.

Actions, executed in order:
{"type":"pick","object":"<object id>"}   grasp and lift. The gripper must be empty and nothing may sit on the object.
{"type":"place","target":"<bin id | object id to stack on | spot>"}   set the held object down. Spots: left, right, center, front, back. Nothing can be stacked on the purple_ball, and an object carries at most one thing on top.
{"type":"point","object":"<object id>"}   hover over an object to show where it is.
{"type":"wave"}
{"type":"home"}
{"type":"gripper","state":"open"|"close"}   only when nothing is held.

Rules:
- Plan the complete sequence that achieves the whole instruction, including any clearing moves it needs first. Never stop after a preparatory step.
- Moving something is a pick followed by a place. Use only ids from the scene.
- To reach an object that has something on top, move the top object first.
- Physics is real: grasps can slip and stacked objects can fall. You are told how your previous plan went; trust the current scene over what you planned.
- If the instruction is ambiguous or impossible, return no actions and ask a short question in "reply".
- "reply" is one short sentence in the operator's language saying what you will do.

Respond with only a JSON object, no prose and no code fence:
{"reasoning": "...", "reply": "...", "actions": [ ... ]}
"reasoning" is your brief working, written first: which objects the instruction refers to and why, with any distances computed from the coordinates."""


def _user_turn(text: str, scene: str, held: str | None) -> str:
    return (f"Objects seen by the camera:\n{scene}\n"
            f"Containers: left_bin (x 0.30, y 0.45) and right_bin (x 0.30, y -0.45); each holds several objects.\n"
            f"Gripper holds: {held or 'nothing'}.\n\nOperator: {text}")


def _parse(raw: str) -> dict:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fence:
        raw = fence.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("the model did not return a JSON plan")
    plan = json.loads(raw[start:end + 1])
    actions = [a for a in plan.get("actions", []) if isinstance(a, dict) and isinstance(a.get("type"), str)]
    return {"reply": str(plan.get("reply", "")), "actions": actions, "reasoning": str(plan.get("reasoning", ""))}


class Brain:
    def __init__(self):
        self.history: list[dict] = []   # recent operator/Astra exchanges, so "it" and "that one" resolve
        self.last_result = ""           # how the previous plan went, told to the model on the next turn

    def note_result(self, text: str):
        self.last_result = text

    def clear(self):
        self.history.clear()
        self.last_result = ""

    def providers(self) -> dict:
        return {
            "claude": {"default_model": DEFAULT_CLAUDE_MODEL,
                       "configured": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))},
            "openai": {"default_model": DEFAULT_OPENAI_MODEL, "configured": bool(os.environ.get("OPENAI_API_KEY")),
                       "base_url": os.environ.get("OPENAI_BASE_URL", "")},
        }

    def plan(self, text: str, scene: str, held: str | None, provider: str, model: str | None,
             base_url: str | None = None, image: bytes | None = None) -> dict:
        """image: the camera's current JPEG frame. Only the current turn carries it; history is text."""
        content = _user_turn(text, scene, held)
        if self.last_result:
            content = f"Outcome of your previous plan: {self.last_result}\n\n{content}"
        img = base64.b64encode(image).decode() if image else None
        if provider == "openai":
            parts = [{"type": "text", "text": content}]
            if img:
                parts.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
            raw = self._openai(self.history[-6:] + [{"role": "user", "content": parts}], model or DEFAULT_OPENAI_MODEL, base_url)
        else:
            parts = [{"type": "text", "text": content}]
            if img:
                parts.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}})
            raw = self._claude(self.history[-6:] + [{"role": "user", "content": parts}], model or DEFAULT_CLAUDE_MODEL)
        plan = _parse(raw)
        prev = f"Outcome of your previous plan: {self.last_result}\n" if self.last_result else ""
        self.history += [{"role": "user", "content": f"{prev}Operator: {text}"},
                         {"role": "assistant", "content": json.dumps(plan)}]
        self.last_result = ""
        return plan

    def _claude(self, messages, model) -> str:
        client = anthropic.Anthropic()
        try:
            resp = client.beta.messages.create(
                model=model,
                max_tokens=16000,
                system=SYSTEM,
                messages=messages,
                output_config={"effort": "low"},   # short plans: low effort keeps the arm responsive
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError:
            raise RuntimeError("Claude rejected the API key. Set ANTHROPIC_API_KEY in .env or run `ant auth login`.")
        except anthropic.NotFoundError:
            raise RuntimeError(f'Claude does not know the model "{model}".')
        except anthropic.RateLimitError:
            raise RuntimeError("Claude rate limit reached. Wait a moment and try again.")
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Claude API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError:
            raise RuntimeError("Could not reach the Claude API. Check the internet connection.")
        except TypeError as e:   # the SDK found no API key, auth token or `ant` profile
            if "authentication" not in str(e):
                raise
            raise RuntimeError("No Claude credentials found. Add ANTHROPIC_API_KEY to .env and restart the server.")
        if resp.stop_reason == "refusal":
            raise RuntimeError("Claude declined this instruction.")
        return "".join(b.text for b in resp.content if b.type == "text")

    def _openai(self, messages, model, base_url) -> str:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("No OPENAI_API_KEY found. Add it to .env and restart the server.")
        client = openai.OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None)
        try:
            resp = client.chat.completions.create(model=model,
                                                  messages=[{"role": "system", "content": SYSTEM}] + messages)
        except openai.AuthenticationError:
            raise RuntimeError("The OpenAI-compatible endpoint rejected the key. Set OPENAI_API_KEY in .env.")
        except openai.NotFoundError:
            raise RuntimeError(f'The endpoint does not know the model "{model}".')
        except openai.RateLimitError:
            raise RuntimeError("Rate limit reached on the OpenAI-compatible endpoint.")
        except openai.APIStatusError as e:
            raise RuntimeError(f"OpenAI-compatible API error {e.status_code}: {e.message}")
        except openai.APIConnectionError:
            raise RuntimeError("Could not reach the OpenAI-compatible endpoint.")
        return resp.choices[0].message.content or ""
