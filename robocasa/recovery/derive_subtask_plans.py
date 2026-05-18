"""Derive recovery-oriented subtask plans for RoboCasa composite tasks.

This is a static analysis tool. It does not claim to recover official ground
truth subtask annotations. Instead, it combines source docstring steps, docs
metadata, skill tags, object configs, fixture hints, and success-predicate calls
into transparent heuristic plans that can seed a recovery monitor.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
COMPOSITE_ROOT = ROOT / "robocasa" / "environments" / "kitchen" / "composite"
TASK_ATTRS = ROOT / "docs" / "composite_tasks" / "task_attributes.json"
DOCS_DROPDOWN = ROOT / "docs" / "composite_tasks" / "composite_tasks_dropdown.js"

_LIB_SPEC = importlib.util.spec_from_file_location(
    "atomic_skill_library", Path(__file__).with_name("atomic_skill_library.py")
)
if _LIB_SPEC is None or _LIB_SPEC.loader is None:
    raise RuntimeError("Could not load atomic skill library")
_LIB_MODULE = importlib.util.module_from_spec(_LIB_SPEC)
_LIB_SPEC.loader.exec_module(_LIB_MODULE)
ATOMIC_SKILL_LIBRARY = _LIB_MODULE.ATOMIC_SKILL_LIBRARY

TAG_TO_SKILL = {
    "door_open": "open_fixture",
    "drawer_open": "open_fixture",
    "blender_lid_open": "open_fixture",
    "door_close": "close_fixture",
    "drawer_close": "close_fixture",
    "stand_mixer_close": "close_fixture",
    "blender_lid_close": "close_fixture",
    "kettle_lid_close": "close_fixture",
    "rack_slide": "slide_rack",
    "knob_twist": "turn_knob",
    "lever_turn": "turn_lever",
    "button_press": "press_button",
    "start_toaster": "turn_lever",
    "PickPlace": "pick_object",
}

SKILL_LABELS = {key: value["label"] for key, value in ATOMIC_SKILL_LIBRARY.items()}

EVAL_COMPOSITE_PLAN_OVERRIDES = {
    "DeliverStraw": [
        ("open_fixture", "Open the front drawer to access the straw.", "drawer"),
        ("pick_object", "Pick the straw from the drawer.", "drawer"),
        (
            "place_object_in_target",
            "Place the straw inside the glass cup on the dining counter.",
            "dining_counter",
        ),
        (
            "release_and_retreat",
            "Release the straw and move the gripper away from the glass.",
            None,
        ),
    ],
    "GetToastedBread": [
        ("turn_lever", "Lower the toaster lever to start toasting.", "toaster"),
        (
            "pick_object",
            "Pick the toasted bread from the toaster after it pops up.",
            "toaster",
        ),
        (
            "place_object_on_target",
            "Place the toasted bread on the plate on the dining counter.",
            "dining_counter",
        ),
        (
            "release_and_retreat",
            "Release the bread and move the gripper away from the plate.",
            None,
        ),
    ],
    "KettleBoiling": [
        ("pick_object", "Pick the kettle from the counter.", "counter"),
        (
            "place_object_on_target",
            "Place the kettle on the target stove burner.",
            "stove",
        ),
        ("turn_knob", "Turn on the target stove burner.", "stove"),
        (
            "release_and_retreat",
            "Move the gripper away from the kettle and stove controls.",
            None,
        ),
    ],
    "LoadDishwasher": [
        (
            "slide_rack",
            "Pull out the dishwasher rack to expose the loading area.",
            "dishwasher",
        ),
        ("pick_object", "Pick the dishes from the counter.", "counter"),
        (
            "place_object_on_target",
            "Place the dishes on the dishwasher rack.",
            "dishwasher",
        ),
        ("close_fixture", "Close the dishwasher door fully.", "dishwasher"),
        (
            "release_and_retreat",
            "Move the gripper away from the closed dishwasher.",
            None,
        ),
    ],
    "PackIdenticalLunches": [
        ("pick_object", "Pick one vegetable from the fridge.", "fridge"),
        (
            "place_object_in_target",
            "Place the vegetable in the first tupperware.",
            "counter",
        ),
        ("pick_object", "Pick one meat item from the fridge.", "fridge"),
        (
            "place_object_in_target",
            "Place the meat item in the first tupperware.",
            "counter",
        ),
        ("pick_object", "Pick the second vegetable from the fridge.", "fridge"),
        (
            "place_object_in_target",
            "Place the second vegetable in the second tupperware.",
            "counter",
        ),
        ("pick_object", "Pick the second meat item from the fridge.", "fridge"),
        (
            "place_object_in_target",
            "Place the second meat item in the second tupperware.",
            "counter",
        ),
        (
            "release_and_retreat",
            "Release the last item and move the gripper away from both tupperwares.",
            None,
        ),
    ],
    "PreSoakPan": [
        ("pick_object", "Pick the pan from the counter.", "counter"),
        ("place_object_in_target", "Place the pan in the sink.", "sink"),
        ("pick_object", "Pick the sponge from the counter.", "counter"),
        ("place_object_in_target", "Place the sponge in the sink.", "sink"),
        ("turn_lever", "Turn on the sink water.", "sink"),
        (
            "release_and_retreat",
            "Move the gripper away from the sink after turning on the water.",
            None,
        ),
    ],
    "PrepareCoffee": [
        ("pick_object", "Pick the mug from the open cabinet.", "cab"),
        (
            "place_object_on_target",
            "Place the mug under the coffee machine dispenser.",
            "coffee_machine",
        ),
        ("press_button", "Press the coffee machine start button.", "coffee_machine"),
        (
            "release_and_retreat",
            "Move the gripper away from the mug and start button.",
            None,
        ),
    ],
    "RinseSinkBasin": [
        ("turn_lever", "Turn on the sink faucet.", "sink"),
        (
            "turn_lever",
            "Move the spout left and right to rinse all locations of the sink basin.",
            "sink",
        ),
    ],
    "ScrubCuttingBoard": [
        ("pick_object", "Pick up the sponge from the counter.", "counter"),
        (
            "place_object_on_target",
            "Press and scrub the sponge on the cutting board.",
            "counter",
        ),
        (
            "release_and_retreat",
            "Release the sponge and move the gripper away from the cutting board.",
            None,
        ),
    ],
    "SearingMeat": [
        ("pick_object", "Pick the pan from the open cabinet.", "cab"),
        (
            "place_object_on_target",
            "Place the pan on the target stove burner.",
            "stove",
        ),
        ("pick_object", "Pick the meat from the counter.", "counter"),
        ("place_object_in_target", "Place the meat in the pan on the stove.", "stove"),
        ("turn_knob", "Turn on the target stove burner.", "stove"),
        (
            "release_and_retreat",
            "Move the gripper away from the pan and stove controls.",
            None,
        ),
    ],
    "SetUpCuttingStation": [
        ("pick_object", "Pick the knife from the open drawer.", "drawer"),
        ("place_object_on_target", "Place the knife on the cutting board.", "counter"),
        ("pick_object", "Pick the meat from the plate.", "counter"),
        ("place_object_on_target", "Place the meat on the cutting board.", "counter"),
        (
            "release_and_retreat",
            "Release the last object and move the gripper away from the cutting board.",
            None,
        ),
    ],
    "StackBowlsCabinet": [
        ("pick_object", "Pick the larger bowl from the counter.", "counter"),
        (
            "place_object_in_target",
            "Place the larger bowl in the open cabinet.",
            "cabinet",
        ),
        ("pick_object", "Pick the smaller bowl from the counter.", "counter"),
        (
            "place_object_on_target",
            "Stack the smaller bowl on top of the larger bowl inside the cabinet.",
            "cabinet",
        ),
        (
            "release_and_retreat",
            "Release the bowl stack and move the gripper away from the cabinet.",
            None,
        ),
    ],
    "SteamInMicrowave": [
        ("pick_object", "Pick the vegetable from the sink.", "sink"),
        ("place_object_in_target", "Place the vegetable in the bowl.", "counter"),
        ("pick_object", "Pick the bowl containing the vegetable.", "counter"),
        ("place_object_in_target", "Place the bowl inside the microwave.", "microwave"),
        ("close_fixture", "Close the microwave door.", "microwave"),
        ("press_button", "Press the microwave start button.", "microwave"),
        (
            "release_and_retreat",
            "Move the gripper away from the bowl and microwave controls.",
            None,
        ),
    ],
    "StirVegetables": [
        ("pick_object", "Pick the first vegetable from the counter.", "counter"),
        ("place_object_in_target", "Place the first vegetable in the pot.", "stove"),
        ("pick_object", "Pick the second vegetable from the counter.", "counter"),
        ("place_object_in_target", "Place the second vegetable in the pot.", "stove"),
        ("pick_object", "Pick the spatula from the counter.", "counter"),
        (
            "place_object_in_target",
            "Move the spatula inside the pot and stir the vegetables.",
            "stove",
        ),
        (
            "release_and_retreat",
            "Release the spatula and move the gripper away from the pot.",
            None,
        ),
    ],
    "StoreLeftoversInBowl": [
        ("pick_object", "Pick the chicken drumstick from its plate.", "dining_counter"),
        (
            "place_object_in_target",
            "Place the chicken drumstick in the bowl.",
            "dining_counter",
        ),
        ("pick_object", "Pick the vegetable from its plate.", "dining_counter"),
        (
            "place_object_in_target",
            "Place the vegetable in the bowl.",
            "dining_counter",
        ),
        ("pick_object", "Pick the bowl containing the leftovers.", "dining_counter"),
        ("place_object_in_target", "Place the bowl in the open fridge.", "fridge"),
        (
            "release_and_retreat",
            "Release the bowl and move the gripper away from the fridge.",
            None,
        ),
    ],
    "WashLettuce": [
        ("pick_object", "Pick the lettuce or colander from the counter.", "counter"),
        (
            "place_object_in_target",
            "Position the lettuce in the sink under the faucet.",
            "sink",
        ),
        (
            "turn_lever",
            "Turn on the sink faucet and keep water running over the lettuce.",
            "sink",
        ),
        (
            "release_and_retreat",
            "Release the lettuce or colander after washing and move the gripper away.",
            None,
        ),
    ],
    "ArrangeBreadBasket": [
        ("open_fixture", "Open the cabinet containing the bread.", "cab"),
        ("pick_object", "Pick the bread from the cabinet.", "cab"),
        ("place_object_in_target", "Place the bread in the basket.", "counter"),
        ("pick_object", "Pick the basket containing the bread.", "counter"),
        (
            "place_object_on_target",
            "Move the basket to the dining counter.",
            "dining_table",
        ),
        (
            "release_and_retreat",
            "Release the basket and move the gripper away from the dining counter.",
            None,
        ),
    ],
    "ArrangeTea": [
        ("pick_object", "Pick the kettle from the counter.", "counter"),
        ("place_object_on_target", "Place the kettle on the tray.", "counter"),
        ("pick_object", "Pick the mug from the open cabinet.", "cab"),
        ("place_object_on_target", "Place the mug on the tray.", "counter"),
        ("close_fixture", "Close the cabinet doors.", "cab"),
        (
            "release_and_retreat",
            "Move the gripper away from the tray and closed cabinet.",
            None,
        ),
    ],
    "BreadSelection": [
        (
            "pick_object",
            "Select and pick the croissant from the pastries on the counter.",
            "counter",
        ),
        (
            "place_object_on_target",
            "Place the croissant on the cutting board.",
            "counter",
        ),
        ("pick_object", "Pick the jam jar from the open cabinet.", "cab"),
        (
            "place_object_on_target",
            "Place the jam jar alongside the croissant on the cutting board.",
            "counter",
        ),
        (
            "release_and_retreat",
            "Release the jam jar and move the gripper away from the cutting board.",
            None,
        ),
    ],
    "CategorizeCondiments": [
        ("pick_object", "Pick the shaker from the counter.", "counter"),
        (
            "place_object_in_target",
            "Place the shaker next to its counterpart in the open cabinet.",
            "cab",
        ),
        ("pick_object", "Pick the condiment bottle from the counter.", "counter"),
        (
            "place_object_in_target",
            "Place the condiment bottle next to its counterpart in the open cabinet.",
            "cab",
        ),
        (
            "release_and_retreat",
            "Release the bottle and move the gripper away from the cabinet.",
            None,
        ),
    ],
    "CuttingToolSelection": [
        ("open_fixture", "Open the drawer containing the cutting tools.", "drawer"),
        (
            "pick_object",
            "Select and pick the appropriate cutting tool for the food skin.",
            "drawer",
        ),
        (
            "place_object_on_target",
            "Place the selected cutting tool on the cutting board.",
            "counter",
        ),
        (
            "release_and_retreat",
            "Release the cutting tool and move the gripper away from the cutting board.",
            None,
        ),
    ],
    "GarnishPancake": [
        ("open_fixture", "Open the fridge to access the strawberry.", "fridge"),
        ("pick_object", "Pick the strawberry from the fridge.", "fridge"),
        (
            "place_object_on_target",
            "Place the strawberry on top of the pancake.",
            "dining_counter",
        ),
        (
            "release_and_retreat",
            "Release the strawberry and move the gripper away from the pancake.",
            None,
        ),
    ],
    "GatherTableware": [
        ("pick_object", "Pick the glass from the second open cabinet.", "cabinet"),
        (
            "place_object_in_target",
            "Place the glass with the other glasses in the target cabinet.",
            "cabinet",
        ),
        ("pick_object", "Pick the bowl from the cabinet.", "cabinet"),
        (
            "place_object_in_target",
            "Place the bowl on the opposite side of the target cabinet from the glasses.",
            "cabinet",
        ),
        (
            "release_and_retreat",
            "Release the tableware and move the gripper away from the cabinet.",
            None,
        ),
    ],
    "HeatKebabSandwich": [
        ("pick_object", "Pick the kebab skewer from the plate.", "counter"),
        (
            "place_object_in_target",
            "Place the kebab skewer inside the open toaster oven.",
            "toaster_oven",
        ),
        ("pick_object", "Pick the baguette bread from the plate.", "counter"),
        (
            "place_object_in_target",
            "Place the baguette bread inside the open toaster oven.",
            "toaster_oven",
        ),
        ("close_fixture", "Close the toaster oven door.", "toaster_oven"),
        ("turn_knob", "Set the toaster oven timer.", "toaster_oven"),
        (
            "release_and_retreat",
            "Move the gripper away from the toaster oven and timer controls.",
            None,
        ),
    ],
    "MakeIceLemonade": [
        ("open_fixture", "Open the fridge to access the lemon wedge.", "fridge"),
        ("pick_object", "Pick the lemon wedge from the fridge.", "fridge"),
        (
            "place_object_in_target",
            "Place the lemon wedge in the glass of lemonade.",
            "counter",
        ),
        ("pick_object", "Pick one ice cube from the ice bowl.", "counter"),
        (
            "place_object_in_target",
            "Place the ice cube in the glass of lemonade.",
            "counter",
        ),
        (
            "release_and_retreat",
            "Release the last object and move the gripper away from the glass.",
            None,
        ),
    ],
    "PanTransfer": [
        ("pick_object", "Pick up the pan containing the vegetables.", "stove"),
        (
            "place_object_on_target",
            "Dump the vegetables from the pan onto the plate.",
            "counter",
        ),
        ("place_object_on_target", "Return the pan to the stove.", "stove"),
        (
            "release_and_retreat",
            "Release the pan and move the gripper away from the stove.",
            None,
        ),
    ],
    "PortionHotDogs": [
        ("pick_object", "Pick one hot dog bun from the bowl.", "dining_counter"),
        (
            "place_object_on_target",
            "Place the bun on the first plate.",
            "dining_counter",
        ),
        ("pick_object", "Pick one sausage from the bowl.", "dining_counter"),
        (
            "place_object_on_target",
            "Place the sausage on the first plate.",
            "dining_counter",
        ),
        ("pick_object", "Pick the second hot dog bun from the bowl.", "dining_counter"),
        (
            "place_object_on_target",
            "Place the second bun on the second plate.",
            "dining_counter",
        ),
        ("pick_object", "Pick the second sausage from the bowl.", "dining_counter"),
        (
            "place_object_on_target",
            "Place the second sausage on the second plate.",
            "dining_counter",
        ),
        (
            "release_and_retreat",
            "Release the last item and move the gripper away from both plates.",
            None,
        ),
    ],
    "RecycleBottlesByType": [
        (
            "pick_object",
            "Pick the plastic bottle from the middle group.",
            "dining_counter",
        ),
        (
            "place_object_on_target",
            "Place the plastic bottle with the plastics group.",
            "dining_counter",
        ),
        (
            "pick_object",
            "Pick the glass bottle from the middle group.",
            "dining_counter",
        ),
        (
            "place_object_on_target",
            "Place the glass bottle with the glass group.",
            "dining_counter",
        ),
        (
            "release_and_retreat",
            "Release the last bottle and move the gripper away from the recycling groups.",
            None,
        ),
    ],
    "SeparateFreezerRack": [
        ("pick_object", "Pick the meat tupperware container.", "counter"),
        (
            "place_object_on_target",
            "Place the meat container on the second highest freezer rack.",
            "freezer",
        ),
        ("pick_object", "Pick the vegetable tupperware container.", "counter"),
        (
            "place_object_on_target",
            "Place the vegetable container on the highest freezer rack.",
            "freezer",
        ),
        (
            "release_and_retreat",
            "Release the vegetable container and move the gripper away from the freezer.",
            None,
        ),
    ],
    "WaffleReheat": [
        ("open_fixture", "Open the microwave door.", "microwave"),
        ("pick_object", "Pick the bowl containing the waffle.", "counter"),
        ("place_object_in_target", "Place the bowl inside the microwave.", "microwave"),
        ("close_fixture", "Close the microwave door.", "microwave"),
        ("press_button", "Turn on the microwave.", "microwave"),
        (
            "release_and_retreat",
            "Move the gripper away from the bowl and microwave controls.",
            None,
        ),
    ],
    "WashFruitColander": [
        ("pick_object", "Pick the colander from the counter.", "counter"),
        ("place_object_in_target", "Place the colander in the sink.", "sink"),
        ("pick_object", "Pick the fruit from the counter.", "counter"),
        ("place_object_in_target", "Place the fruit in the colander.", "sink"),
        (
            "turn_lever",
            "Turn on the sink faucet and pour water over the colander.",
            "sink",
        ),
        (
            "release_and_retreat",
            "Move the gripper away from the colander after rinsing.",
            None,
        ),
    ],
    "WeighIngredients": [
        ("pick_object", "Pick the packaged food from the open cabinet.", "cab"),
        (
            "place_object_on_target",
            "Place the packaged food on the digital scale.",
            "counter",
        ),
        ("close_fixture", "Close the cabinet.", "cab"),
        (
            "release_and_retreat",
            "Move the gripper away from the scale and closed cabinet.",
            None,
        ),
    ],
}

FIXTURE_WORDS = [
    "fridge",
    "freezer",
    "cabinet",
    "drawer",
    "dishwasher",
    "oven",
    "microwave",
    "toaster oven",
    "toaster",
    "blender",
    "stand mixer",
    "sink",
    "stove",
    "counter",
]


def camel_to_words(name: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").lower()


def load_task_attributes() -> list[dict[str, Any]]:
    data = json.loads(TASK_ATTRS.read_text())
    return data["tasks"]


def class_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in COMPOSITE_ROOT.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                out[node.name] = path
    return out


def task_tag_maps() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    js = DOCS_DROPDOWN.read_text()

    def parse_map(name: str) -> dict[str, list[str]]:
        match = re.search(rf"const {name} = new Map\(\[(.*?)\]\);", js, flags=re.S)
        if not match:
            return {}
        body = match.group(1)
        result: dict[str, list[str]] = {}
        for task, tags in re.findall(r'\["([^"]+)",\s*\[(.*?)\]\]', body, flags=re.S):
            result[task] = re.findall(r'"([^"]+)"', tags)
        return result

    return parse_map("TASK_TAGS"), parse_map("TASK_TAG_REMOVALS")


def infer_skill_tags(
    task_name: str,
    mobile: str | None,
    description: str,
    tags_map: dict[str, list[str]],
    removals: dict[str, list[str]],
) -> list[str]:
    tags = list(tags_map.get(task_name, ["PickPlace"]))
    if mobile == "Yes" and "nav" not in tags:
        tags.append("nav")

    text = f"{camel_to_words(task_name)} {description}".lower()
    if "open" in text:
        if "drawer" in text:
            tags.append("drawer_open")
        elif any(
            word in text
            for word in [
                "door",
                "cabinet",
                "fridge",
                "freezer",
                "dishwasher",
                "oven",
                "microwave",
            ]
        ):
            tags.append("door_open")
    if re.search(r"\b(close|shut)\b", text):
        if "drawer" in text:
            tags.append("drawer_close")
        elif any(
            word in text
            for word in [
                "door",
                "cabinet",
                "fridge",
                "freezer",
                "dishwasher",
                "oven",
                "microwave",
            ]
        ):
            tags.append("door_close")
    if re.search(r"\b(press|push|button)\b", text):
        tags.append("button_press")
    if re.search(r"\b(slide|rack|tray)\b", text):
        tags.append("rack_slide")
    if re.search(
        r"\b(turn|twist|rotate|adjust|preheat|temperature|burner|knob)\b", text
    ):
        tags.append("knob_twist")
    if re.search(r"\b(lever|faucet|spout|water)\b", text):
        tags.append("lever_turn")

    for tag in removals.get(task_name, []):
        tags = [item for item in tags if item != tag]

    ordered = []
    for tag in tags:
        if tag not in ordered:
            ordered.append(tag)
    return ordered


def get_class_node(path: Path, class_name: str) -> ast.ClassDef | None:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def method_node(class_node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def extract_docstring_steps(class_node: ast.ClassDef) -> list[str]:
    doc = ast.get_docstring(class_node) or ""
    match = re.search(r"Steps:\s*(.*)", doc, flags=re.S)
    if not match:
        return []
    steps_text = match.group(1)
    steps: list[str] = []
    current: list[str] = []
    for raw_line in steps_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        number = re.match(r"^\d+\.\s*(.*)", line)
        if number:
            if current:
                steps.append(" ".join(current).strip())
            current = [number.group(1).strip()]
        elif current and not re.match(r"^[A-Z][A-Za-z ]+:$", line):
            current.append(line)
    if current:
        steps.append(" ".join(current).strip())
    return steps


def extract_string_literals(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    values: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            text = " ".join(sub.value.split())
            if len(text) > 12:
                values.append(text)
    return values


def extract_object_names(source: str) -> list[str]:
    names = re.findall(r'name\s*=\s*"([^"]+)"', source)
    names += re.findall(r'"name"\s*:\s*"([^"]+)"', source)
    return sorted(set(names))


def extract_fixture_names(source: str) -> list[str]:
    names = re.findall(r'register_fixture_ref\(\s*"([^"]+)"', source)
    for word in FIXTURE_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", source, flags=re.I):
            names.append(word)
    return sorted(set(names))


def extract_setup_state_hints(source: str) -> list[str]:
    hints = []
    for fixture, action in re.findall(
        r"self\.([A-Za-z_][A-Za-z0-9_]*)\.(open_door|close_door)\(", source
    ):
        hints.append(f"{fixture}.{action}")
    for line in source.splitlines():
        if any(
            token in line
            for token in ["open_", "close_", "turn_on", "turn_off", "set_"]
        ):
            clean = line.strip()
            if clean and len(clean) < 140:
                hints.append(clean)
    return sorted(set(hints))


def extract_success_hints(source: str) -> list[str]:
    fn_names = []
    for match in re.findall(r"OU\.([A-Za-z_][A-Za-z0-9_]*)\(", source):
        fn_names.append(f"OU.{match}")
    for match in re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)\(", source):
        if "check" in match or "success" in match:
            fn_names.append(f"self.{match}")
    vars_ = re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", source, flags=re.M)
    useful_vars = [
        v
        for v in vars_
        if any(
            token in v
            for token in [
                "in_",
                "_in",
                "far",
                "open",
                "closed",
                "on_",
                "_on",
                "near",
                "touch",
                "cluster",
                "success",
            ]
        )
    ]
    return sorted(set(fn_names + useful_vars))


def classify_text(text: str) -> list[str]:
    lower = text.lower()
    skills: list[str] = []
    if re.search(r"\b(open)\b", lower):
        skills.append("open_fixture")
    if re.search(r"\b(close|shut)\b", lower):
        skills.append("close_fixture")
    if re.search(r"\b(pick|grab|take|retrieve|gather|move|place|put|add|set)\b", lower):
        skills.append("pick_object")
        if re.search(r"\b(in|into|inside)\b", lower):
            skills.append("place_object_in_target")
        elif re.search(r"\b(on|onto|over)\b", lower):
            skills.append("place_object_on_target")
        else:
            skills.append("place_object_on_target")
    if re.search(r"\b(slide|pull out|push in|rack|tray)\b", lower):
        skills.append("slide_rack")
    if re.search(
        r"\b(turn|twist|rotate|adjust|preheat|temperature|burner|knob)\b", lower
    ):
        skills.append("turn_knob")
    if re.search(r"\b(lever|faucet|spout|water)\b", lower):
        skills.append("turn_lever")
    if re.search(r"\b(press|push|button|start|stop)\b", lower):
        skills.append("press_button")
    return dedupe(skills)


def dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def infer_fixture(text: str, fixtures: list[str]) -> str | None:
    lower = text.lower()
    for fixture in fixtures:
        if fixture.replace("_", " ") in lower:
            return fixture
    for word in FIXTURE_WORDS:
        if word in lower:
            return word
    return fixtures[0] if fixtures else None


def split_instruction(description: str) -> list[str]:
    parts = []
    for chunk in re.split(r"\.\s+|\bThen\b", description):
        clean = chunk.strip(" .")
        if clean:
            parts.append(clean + ".")
    return parts or [description]


def infer_plan(
    task: dict[str, Any],
    doc_steps: list[str],
    skill_tags: list[str],
    fixtures: list[str],
    success_hints: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    plan: list[dict[str, Any]] = []
    gaps: list[str] = []

    def add(skill_id: str, source: str, text: str, fixture: str | None = None) -> None:
        if skill_id not in ATOMIC_SKILL_LIBRARY:
            return
        step = {
            "index": len(plan) + 1,
            "skill_id": skill_id,
            "skill_label": SKILL_LABELS[skill_id],
            "source": source,
            "text": text,
        }
        if fixture:
            step["fixture_hint"] = fixture
        plan.append(step)

    override = EVAL_COMPOSITE_PLAN_OVERRIDES.get(task["name"])
    if override:
        for skill_id, text, fixture in override:
            add(skill_id, "curated_eval_override", text, fixture)
        return plan, gaps

    base_steps = doc_steps or split_instruction(task["description"])
    if not doc_steps:
        gaps.append(
            "No source docstring Steps: block found; plan inferred from task description only."
        )
    for source_step in base_steps:
        for skill_id in classify_text(source_step):
            add(
                skill_id,
                "docstring_steps" if doc_steps else "task_description",
                source_step,
                infer_fixture(source_step, fixtures),
            )

    has_skill = {step["skill_id"] for step in plan}
    prefix_steps: list[tuple[str, str, str]] = []
    suffix_steps: list[tuple[str, str, str]] = []
    if (
        "door_open" in skill_tags
        or "drawer_open" in skill_tags
        or "blender_lid_open" in skill_tags
    ) and "open_fixture" not in has_skill:
        prefix_steps.append(
            (
                "open_fixture",
                "Open required fixture before object manipulation.",
                "docs_skill_tag",
            )
        )
    if "rack_slide" in skill_tags and "slide_rack" not in has_skill:
        prefix_steps.append(
            (
                "slide_rack",
                "Slide rack or tray to expose target workspace.",
                "docs_skill_tag",
            )
        )
    if "knob_twist" in skill_tags and "turn_knob" not in has_skill:
        suffix_steps.append(
            ("turn_knob", "Set required appliance knob or dial.", "docs_skill_tag")
        )
    if "lever_turn" in skill_tags and "turn_lever" not in has_skill:
        suffix_steps.append(
            ("turn_lever", "Set required lever, faucet, or spout.", "docs_skill_tag")
        )
    if "button_press" in skill_tags and "press_button" not in has_skill:
        suffix_steps.append(
            ("press_button", "Press required appliance button.", "docs_skill_tag")
        )

    if prefix_steps:
        old_plan = plan
        plan = []
        for skill_id, text, source in prefix_steps:
            add(skill_id, source, text, infer_fixture(task["description"], fixtures))
        for step in old_plan:
            step["index"] = len(plan) + 1
            plan.append(step)

    for skill_id, text, source in suffix_steps:
        add(skill_id, source, text, infer_fixture(task["description"], fixtures))

    if (
        "door_close" in skill_tags
        or "drawer_close" in skill_tags
        or "stand_mixer_close" in skill_tags
        or "blender_lid_close" in skill_tags
        or "kettle_lid_close" in skill_tags
    ):
        add(
            "close_fixture",
            "docs_skill_tag",
            "Close required fixture after manipulation.",
            infer_fixture(task["description"], fixtures),
        )

    if any("gripper_obj_far" in hint or "far" in hint for hint in success_hints) or any(
        step["skill_id"].startswith("place_") for step in plan
    ):
        add(
            "release_and_retreat",
            "success_predicate",
            "Release manipulated objects and retreat until gripper-far predicates are satisfied.",
        )

    # Remove exact duplicate adjacent operations from mixed docs/tag evidence.
    compact: list[dict[str, Any]] = []
    for step in plan:
        if (
            compact
            and compact[-1]["skill_id"] == step["skill_id"]
            and compact[-1].get("fixture_hint") == step.get("fixture_hint")
        ):
            compact[-1]["source"] += f"+{step['source']}"
            continue
        step["index"] = len(compact) + 1
        compact.append(step)

    declared = int(task.get("num_subtasks") or 0)
    if declared and abs(len(compact) - declared) > max(2, declared // 2):
        gaps.append(
            f"Derived {len(compact)} skill steps differs strongly from declared num_subtasks={declared}."
        )

    return compact, gaps


def derive_plans() -> dict[str, Any]:
    tasks = [
        task for task in load_task_attributes() if task.get("activity") != "Atomic"
    ]
    index = class_index()
    tag_map, removals = task_tag_maps()

    records: list[dict[str, Any]] = []
    for task in tasks:
        task_name = task["name"]
        path = index.get(task_name)
        source = path.read_text() if path and path.exists() else ""
        class_node = get_class_node(path, task_name) if path else None
        get_ep_meta = method_node(class_node, "get_ep_meta") if class_node else None
        check_success = (
            method_node(class_node, "_check_success") if class_node else None
        )
        obj_cfgs = method_node(class_node, "_get_obj_cfgs") if class_node else None
        setup_scene = method_node(class_node, "_setup_scene") if class_node else None
        setup_refs = (
            method_node(class_node, "_setup_kitchen_references") if class_node else None
        )

        doc_steps = extract_docstring_steps(class_node) if class_node else []
        skill_tags = infer_skill_tags(
            task_name, task.get("moma_required"), task["description"], tag_map, removals
        )
        snippets = "\n".join(
            ast.get_source_segment(source, node) or ""
            for node in [setup_refs, setup_scene, obj_cfgs, check_success]
            if node is not None
        )
        fixtures = extract_fixture_names(snippets)
        success_source = (
            ast.get_source_segment(source, check_success)
            if check_success is not None
            else ""
        )
        success_hints = extract_success_hints(success_source or "")
        plan, gaps = infer_plan(task, doc_steps, skill_tags, fixtures, success_hints)

        if path is None:
            gaps.append("No matching composite task source class found.")

        confidence = (
            "high"
            if doc_steps and success_hints
            else "medium"
            if doc_steps or success_hints
            else "low"
        )
        records.append(
            {
                "task": task_name,
                "activity": task.get("activity"),
                "description": task.get("description"),
                "declared_num_subtasks": task.get("num_subtasks"),
                "moma_required": task.get("moma_required"),
                "source_file": str(path.relative_to(ROOT)) if path else None,
                "episode_language_literals": extract_string_literals(get_ep_meta)[:8],
                "docstring_steps": doc_steps,
                "docs_skill_tags": skill_tags,
                "object_hints": extract_object_names(snippets),
                "fixture_hints": fixtures,
                "setup_state_hints": extract_setup_state_hints(
                    ast.get_source_segment(source, setup_scene) or ""
                )
                if setup_scene
                else [],
                "success_predicate_hints": success_hints,
                "derived_plan": plan,
                "derived_step_count": len(plan),
                "confidence": confidence,
                "gaps": gaps,
            }
        )

    return {
        "metadata": {
            "task_count": len(records),
            "source": "Heuristic static derivation from RoboCasa source/docs. Not official subtask ground truth.",
            "atomic_skill_ids": list(ATOMIC_SKILL_LIBRARY.keys()),
        },
        "atomic_skill_library": ATOMIC_SKILL_LIBRARY,
        "plans": records,
    }


def write_summary(data: dict[str, Any], path: Path) -> None:
    plans = data["plans"]
    counts = {
        "high": sum(1 for plan in plans if plan["confidence"] == "high"),
        "medium": sum(1 for plan in plans if plan["confidence"] == "medium"),
        "low": sum(1 for plan in plans if plan["confidence"] == "low"),
    }
    lines = [
        "# Derived RoboCasa Composite Subtask Plans",
        "",
        "This file summarizes heuristic recovery-oriented plans. These are not official RoboCasa subtask annotations.",
        "",
        f"- Composite tasks processed: {len(plans)}",
        f"- Confidence: high={counts['high']}, medium={counts['medium']}, low={counts['low']}",
        f"- Atomic skill templates: {', '.join(data['metadata']['atomic_skill_ids'])}",
        "",
        "## Example Plans",
        "",
    ]
    for task_name in [
        "MakeIceLemonade",
        "LoadFridgeByType",
        "RecycleBottlesByType",
        "PrepareCoffee",
        "StackBowlsCabinet",
    ]:
        plan = next((item for item in plans if item["task"] == task_name), None)
        if not plan:
            continue
        lines += [
            f"### {task_name}",
            "",
            f"- Description: {plan['description']}",
            f"- Declared `num_subtasks`: {plan['declared_num_subtasks']}",
            f"- Docs skill tags: {', '.join(plan['docs_skill_tags']) or 'none'}",
            f"- Source: `{plan['source_file']}`",
            f"- Confidence: {plan['confidence']}",
            "",
            "| # | Skill | Evidence text | Source |",
            "|---:|---|---|---|",
        ]
        for step in plan["derived_plan"]:
            text = str(step["text"]).replace("|", "\\|")
            lines.append(
                f"| {step['index']} | `{step['skill_id']}` | {text} | {step['source']} |"
            )
        if plan["gaps"]:
            lines += ["", "Gaps:", *[f"- {gap}" for gap in plan["gaps"]]]
        lines.append("")
    path.write_text("\n".join(lines))


def load_target_composite_splits() -> dict[str, list[str]]:
    text = (ROOT / "robocasa" / "utils" / "dataset_registry.py").read_text()
    match = re.search(r"TARGET_TASKS = dict\((.*?)\n\)\n\nLIFELONG", text, re.S)
    if not match:
        raise RuntimeError("Could not locate TARGET_TASKS in dataset_registry.py")

    splits: dict[str, list[str]] = {}
    for split in ["composite_seen", "composite_unseen"]:
        split_match = re.search(rf"{split}=\[(.*?)\]", match.group(1), re.S)
        if not split_match:
            raise RuntimeError(f"Could not locate TARGET_TASKS[{split}]")
        splits[split] = re.findall(r'"([A-Za-z0-9_]+)"', split_match.group(1))
    return splits


def write_eval_composite_csv(data: dict[str, Any], path: Path) -> None:
    plans_by_task = {plan["task"]: plan for plan in data["plans"]}
    splits = load_target_composite_splits()
    fieldnames = [
        "eval_split",
        "task",
        "activity",
        "description",
        "declared_num_subtasks",
        "derived_step_count",
        "confidence",
        "subtask_index",
        "skill_id",
        "skill_label",
        "subtask_text",
        "subtask_source",
        "fixture_hint",
        "object_hints",
        "docs_skill_tags",
        "success_predicate_hints",
        "source_file",
        "gaps",
    ]

    rows = []
    for split, task_names in splits.items():
        for task_name in task_names:
            plan = plans_by_task[task_name]
            for subtask in plan["derived_plan"]:
                rows.append(
                    {
                        "eval_split": split,
                        "task": task_name,
                        "activity": plan.get("activity", ""),
                        "description": plan.get("description", ""),
                        "declared_num_subtasks": plan.get("declared_num_subtasks", ""),
                        "derived_step_count": plan.get("derived_step_count", ""),
                        "confidence": plan.get("confidence", ""),
                        "subtask_index": subtask.get("index", ""),
                        "skill_id": subtask.get("skill_id", ""),
                        "skill_label": subtask.get("skill_label", ""),
                        "subtask_text": subtask.get("text", ""),
                        "subtask_source": subtask.get("source", ""),
                        "fixture_hint": subtask.get("fixture_hint", ""),
                        "object_hints": "; ".join(plan.get("object_hints", [])),
                        "docs_skill_tags": "; ".join(plan.get("docs_skill_tags", [])),
                        "success_predicate_hints": "; ".join(
                            plan.get("success_predicate_hints", [])
                        ),
                        "source_file": plan.get("source_file", ""),
                        "gaps": "; ".join(plan.get("gaps", [])),
                    }
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--eval-composite-csv", type=Path)
    args = parser.parse_args()

    data = derive_plans()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(data, indent=2, sort_keys=True))
    write_summary(data, args.out_summary)
    result = {
        "plans": len(data["plans"]),
        "out_json": str(args.out_json),
        "out_summary": str(args.out_summary),
    }
    if args.eval_composite_csv:
        write_eval_composite_csv(data, args.eval_composite_csv)
        result["eval_composite_csv"] = str(args.eval_composite_csv)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
