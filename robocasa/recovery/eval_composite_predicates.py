"""Runtime subtask predicates for target atomic and composite eval tasks.

These predicates are a monitoring layer only. They decompose task progress for
debugging and recovery, but they do not replace each task's official
``_check_success()`` evaluator.
"""

from __future__ import annotations

import numpy as np


_OBJECT_LABELS = {
    "obj": "the object",
    "obj1": "the first object",
    "obj2": "the second object",
    "veg1": "the first vegetable",
    "veg2": "the second vegetable",
    "cab": "the cabinet",
    "cabinet": "the cabinet",
    "counter": "the counter",
    "receptacle": "the cutting board",
    "container": "the container",
    "food_container": "the cutting board",
    "glass_cup": "the glass",
    "pancake_container": "the plate",
    "vegetable_container": "the pan",
    "waffle_container": "the bowl",
    "meat_tupperware": "the meat container",
    "veg_tupperware": "the vegetable container",
    "digital_scale": "the scale",
    "baguette": "the baguette",
    "basket": "the basket",
    "bowl": "the bowl",
    "bread": "the bread",
    "chicken": "the chicken",
    "chicken_drumstick": "the chicken drumstick",
    "colander": "the colander",
    "croissant": "the croissant",
    "fruit": "the fruit",
    "glass": "the glass",
    "ice": "an ice cube",
    "ice_cube1": "the first ice cube",
    "ice_cube2": "the second ice cube",
    "jam": "the jam",
    "kebab": "the kebab",
    "kettle": "the kettle",
    "knife": "the knife",
    "lemon": "the lemon wedge",
    "lemon_wedge": "the lemon wedge",
    "lettuce": "the lettuce",
    "meat": "the meat",
    "microwave": "the microwave",
    "mug": "the mug",
    "pan": "the pan",
    "pancake": "the pancake",
    "plate": "the plate",
    "sponge": "the sponge",
    "spatula": "the spatula",
    "straw": "the straw",
    "strawberry": "the strawberry",
    "tupperware0": "the first tupperware",
    "tupperware1": "the second tupperware",
    "vegetable": "the vegetable",
    "waffle": "the waffle",
}


_PREDICATE_DESCRIPTION_OVERRIDES = {
    "board_contact_count_reached": "Press sponge on cutting board.",
    "board_sweep_range_reached": "Move sponge across cutting board.",
    "objects_not_duplicated": "Distribute objects.",
    "wrong_tool_not_on_cutting_board": "Remove wrong tool.",
    "robot_did_not_touch_food": "Avoid touching food.",
    "washed_time_reached": "Rinse object.",
    "vegetables_stirred": "Stir vegetables.",
    "bottles_on_table": "Move bottles to table.",
    "bowl_separated_from_glasses": "Separate bowl from glasses.",
    "glasses_clustered": "Group glasses.",
    "plastic_bottles_clustered": "Group plastic bottles.",
    "glass_bottles_clustered": "Group glass bottles.",
    "cabinets_open": "Open cabinets.",
    "dishes_on_rack": "Move dishes to rack.",
    "dishwasher_rack_accessible": "Open dishwasher.",
    "left_basin_rinsed": "Rinse left basin.",
    "center_basin_rinsed": "Rinse center basin.",
    "right_basin_rinsed": "Rinse right basin.",
    "fruit_in_colander": "Move fruit to colander.",
    "colander_under_water": "Move colander under water.",
    "lettuce_under_water": "Move lettuce under water.",
    "water_on": "Turn on water.",
    "burner_on": "Turn on burner.",
    "timer_set": "Set timer.",
    "coffee_started": "Start coffee machine.",
    "microwave_started": "Start microwave.",
    "toaster_started": "Start toaster.",
    "freezer_open": "Open freezer.",
    "fridge_open": "Open fridge.",
    "drawer_open": "Open drawer.",
    "cabinet_open": "Open cabinet.",
    "cabinet_closed": "Close cabinet.",
    "dishwasher_closed": "Close dishwasher.",
    "microwave_closed": "Close microwave.",
    "toaster_oven_closed": "Close toaster oven.",
    "gripper_released": "Release object.",
    "at_target_position": "Move robot to target.",
    "facing_target": "Face target.",
    "object_at_target_and_released": "Move object to target and release it.",
    "final_placement_valid": "Complete final placement.",
}


_TASK_PREDICATE_DESCRIPTION_OVERRIDES = {
    "DeliverStraw": {
        "drawer_open": "Open drawer.",
        "straw_grasped": "Pick straw.",
        "straw_in_glass": "Move straw to glass.",
        "gripper_released": "Release straw.",
    },
    "GetToastedBread": {
        "toaster_started": "Start toaster.",
        "bread_grasped": "Pick bread.",
        "bread_on_plate": "Move bread to plate.",
        "gripper_released": "Release bread.",
    },
    "KettleBoiling": {
        "kettle_grasped": "Pick kettle.",
        "kettle_on_burner": "Move kettle to stove burner.",
        "burner_on": "Turn on burner.",
        "gripper_released": "Release kettle.",
    },
    "LoadDishwasher": {
        "dishwasher_rack_accessible": "Open dishwasher.",
        "dishes_grasped": "Pick dishes.",
        "dishes_on_rack": "Move dishes to dishwasher rack.",
        "dishwasher_closed": "Close dishwasher.",
    },
    "MakeIceLemonade": {
        "fridge_open": "Open fridge.",
        "lemon_grasped": "Pick lemon wedge.",
        "ice_cube1_grasped": "Pick first ice cube.",
        "ice_cube2_grasped": "Pick second ice cube.",
        "lemon_in_glass": "Move lemon wedge to glass.",
        "ice_in_glass": "Move ice cube to glass.",
        "gripper_released": "Release object.",
    },
    "PackIdenticalLunches": {
        "fridge_open": "Open fridge.",
        "tupperware0_has_one_vegetable": "Move vegetable to first tupperware.",
        "tupperware0_has_one_meat": "Move meat to first tupperware.",
        "tupperware1_has_one_vegetable": "Move vegetable to second tupperware.",
        "tupperware1_has_one_meat": "Move meat to second tupperware.",
        "objects_not_duplicated": "Distribute objects.",
        "gripper_released": "Release object.",
    },
    "PreSoakPan": {
        "pan_grasped": "Pick pan.",
        "pan_in_sink": "Move pan to sink.",
        "sponge_grasped": "Pick sponge.",
        "sponge_in_sink": "Move sponge to sink.",
        "water_on": "Turn on water.",
        "gripper_released": "Release objects.",
    },
    "PrepareCoffee": {
        "mug_grasped": "Pick mug.",
        "mug_under_dispenser": "Move mug under coffee dispenser.",
        "coffee_started": "Start coffee machine.",
        "gripper_released": "Release mug.",
    },
    "RinseSinkBasin": {
        "water_on": "Turn on water.",
        "left_basin_rinsed": "Rinse left basin.",
        "center_basin_rinsed": "Rinse center basin.",
        "right_basin_rinsed": "Rinse right basin.",
    },
    "ScrubCuttingBoard": {
        "sponge_grasped": "Pick sponge.",
        "board_contact_count_reached": "Press sponge on cutting board.",
        "board_sweep_range_reached": "Move sponge across cutting board.",
        "gripper_released": "Release sponge.",
    },
    "SearingMeat": {
        "cabinet_open": "Open cabinet.",
        "pan_grasped": "Pick pan.",
        "pan_on_target_burner": "Move pan to target burner.",
        "meat_grasped": "Pick meat.",
        "meat_in_pan": "Move meat to pan.",
        "burner_on": "Turn on burner.",
        "gripper_released": "Release meat.",
    },
    "SetUpCuttingStation": {
        "drawer_open": "Open drawer.",
        "knife_grasped": "Pick knife.",
        "knife_on_cutting_board": "Move knife to cutting board.",
        "meat_grasped": "Pick meat.",
        "meat_on_cutting_board": "Move meat to cutting board.",
        "gripper_released": "Release object.",
    },
    "StackBowlsCabinet": {
        "cabinet_open": "Open cabinet.",
        "larger_bowl_in_cabinet": "Move larger bowl to cabinet.",
        "smaller_bowl_in_cabinet": "Move smaller bowl to cabinet.",
        "bowls_stacked": "Stack bowls.",
        "gripper_released": "Release bowls.",
    },
    "SteamInMicrowave": {
        "vegetable_in_bowl": "Move vegetable to bowl.",
        "bowl_in_microwave": "Move bowl to microwave.",
        "microwave_closed": "Close microwave.",
        "microwave_started": "Start microwave.",
    },
    "StirVegetables": {
        "vegetable1_in_pot": "Move first vegetable to pot.",
        "vegetable2_in_pot": "Move second vegetable to pot.",
        "spatula_grasped": "Pick spatula.",
        "vegetables_stirred": "Stir vegetables.",
        "spatula_released": "Release spatula.",
    },
    "StoreLeftoversInBowl": {
        "chicken_in_bowl": "Move chicken to bowl.",
        "vegetable_in_bowl": "Move vegetable to bowl.",
        "bowl_in_fridge": "Move bowl to fridge.",
        "gripper_released": "Release bowl.",
    },
    "WashLettuce": {
        "water_on": "Turn on water.",
        "lettuce_under_water": "Move lettuce under water.",
        "washed_time_reached": "Rinse lettuce.",
    },
    "ArrangeBreadBasket": {
        "cabinet_open": "Open cabinet.",
        "bread_in_basket": "Move bread to basket.",
        "basket_on_dining_counter": "Move basket to dining counter.",
        "gripper_released": "Release basket.",
    },
    "ArrangeTea": {
        "kettle_on_tray": "Move kettle to tray.",
        "mug_on_tray": "Move mug to tray.",
        "cabinet_closed": "Close cabinet.",
        "gripper_released": "Release object.",
    },
    "BreadSelection": {
        "croissant_on_cutting_board": "Move croissant to cutting board.",
        "jam_on_cutting_board": "Move jam to cutting board.",
        "gripper_released": "Release object.",
    },
    "CategorizeCondiments": {
        "bottle_in_cabinet": "Move bottle to cabinet.",
        "shaker_in_cabinet": "Move shaker to cabinet.",
        "bottle_next_to_counterpart": "Move bottle next to matching bottle.",
        "shaker_next_to_counterpart": "Move shaker next to matching shaker.",
        "gripper_released": "Release objects.",
    },
    "CuttingToolSelection": {
        "drawer_open": "Open drawer.",
        "correct_tool_grasped": "Pick correct tool.",
        "correct_tool_on_cutting_board": "Move correct tool to cutting board.",
        "wrong_tool_not_on_cutting_board": "Remove wrong tool.",
        "gripper_released": "Release object.",
    },
    "GarnishPancake": {
        "fridge_open": "Open fridge.",
        "strawberry_grasped": "Pick strawberry.",
        "pancake_on_plate": "Move pancake to plate.",
        "plate_on_table": "Move plate to table.",
        "strawberry_on_pancake": "Move strawberry to pancake.",
        "gripper_released": "Release strawberry.",
    },
    "GatherTableware": {
        "cabinets_open": "Open cabinets.",
        "glasses_clustered": "Group glasses.",
        "bowl_separated_from_glasses": "Separate bowl from glasses.",
        "gripper_released": "Release tableware.",
    },
    "HeatKebabSandwich": {
        "kebab_in_toaster_oven": "Move kebab to toaster oven.",
        "baguette_in_toaster_oven": "Move baguette to toaster oven.",
        "toaster_oven_closed": "Close toaster oven.",
        "timer_set": "Set timer.",
    },
    "PanTransfer": {
        "pan_grasped": "Pick pan.",
        "vegetable_on_plate": "Move vegetable to plate.",
        "pan_on_stove": "Move pan to stove.",
        "robot_did_not_touch_food": "Avoid touching food.",
        "gripper_released": "Release pan.",
    },
    "PortionHotDogs": {
        "plate1_has_one_bun": "Move bun to first plate.",
        "plate1_has_one_sausage": "Move sausage to first plate.",
        "plate2_has_one_bun": "Move bun to second plate.",
        "plate2_has_one_sausage": "Move sausage to second plate.",
        "gripper_released": "Release food.",
    },
    "RecycleBottlesByType": {
        "plastic_bottles_clustered": "Group plastic bottles.",
        "glass_bottles_clustered": "Group glass bottles.",
        "bottles_on_table": "Move bottles to table.",
        "gripper_released": "Release bottles.",
    },
    "SeparateFreezerRack": {
        "freezer_open": "Open freezer.",
        "meat_in_tupperware": "Move meat to meat container.",
        "vegetables_in_tupperware": "Move vegetables to vegetable container.",
        "meat_container_on_second_rack": "Move meat container to second freezer rack.",
        "vegetable_container_on_top_rack": "Move vegetable container to top freezer rack.",
        "gripper_released": "Release containers.",
    },
    "WaffleReheat": {
        "microwave_open": "Open microwave.",
        "waffle_in_bowl": "Move waffle to bowl.",
        "bowl_in_microwave": "Move bowl to microwave.",
        "microwave_closed": "Close microwave.",
        "microwave_started": "Start microwave.",
    },
    "WashFruitColander": {
        "colander_in_sink": "Move colander to sink.",
        "fruit_in_colander": "Move fruit to colander.",
        "colander_under_water": "Move colander under water.",
    },
    "WeighIngredients": {
        "packaged_food_grasped": "Pick packaged food.",
        "packaged_food_on_scale": "Move packaged food to scale.",
        "packaged_food_upright": "Orient packaged food upright.",
        "cabinet_closed": "Close cabinet.",
        "gripper_released": "Release packaged food.",
    },
    "CloseBlenderLid": {
        "blender_lid_on_blender": "Move blender lid onto blender.",
        "gripper_released": "Release blender lid.",
    },
    "CloseFridge": {
        "fridge_closed": "Close fridge.",
    },
    "CloseToasterOvenDoor": {
        "toaster_oven_closed": "Close toaster oven.",
    },
    "CoffeeSetupMug": {
        "mug_grasped": "Pick mug.",
        "mug_under_dispenser": "Move mug under coffee dispenser.",
        "gripper_released": "Release mug.",
    },
    "NavigateKitchen": {
        "at_target_position": "Move robot to target.",
        "facing_target": "Face target.",
    },
    "OpenCabinet": {
        "cabinet_open": "Open cabinet.",
    },
    "OpenDrawer": {
        "drawer_open": "Open drawer.",
    },
    "OpenStandMixerHead": {
        "stand_mixer_head_open": "Open stand mixer head.",
    },
    "PickPlaceCounterToCabinet": {
        "object_grasped": "Pick object.",
        "object_in_cabinet": "Move object to cabinet.",
        "final_placement_valid": "Keep object in cabinet and release it.",
        "gripper_released": "Release object.",
    },
    "PickPlaceCounterToStove": {
        "object_grasped": "Pick object.",
        "object_in_pan": "Move object to pan.",
        "final_placement_valid": "Keep object in pan and release it.",
        "gripper_released": "Release object.",
    },
    "PickPlaceDrawerToCounter": {
        "object_grasped": "Pick the target object from the drawer.",
        "object_on_counter": "Move the target object from the drawer to the counter.",
        "final_placement_valid": "Keep the target object from the drawer on the counter and release it.",
        "gripper_released": "Release the target object from the drawer.",
    },
    "PickPlaceSinkToCounter": {
        "object_grasped": "Pick object.",
        "object_in_container": "Move object to container.",
        "container_on_counter": "Move container to counter.",
        "final_placement_valid": "Keep object in container on counter and release it.",
        "gripper_released": "Release object.",
    },
    "PickPlaceToasterToCounter": {
        "object_grasped": "Pick object.",
        "object_on_plate": "Move object to plate.",
        "final_placement_valid": "Keep object on plate and release it.",
        "gripper_released": "Release object.",
    },
    "SlideDishwasherRack": {
        "dishwasher_rack_slid": "Slide dishwasher rack.",
    },
    "TurnOffStove": {
        "burner_off": "Turn off burner.",
    },
    "TurnOnElectricKettle": {
        "electric_kettle_on": "Turn on electric kettle.",
    },
    "TurnOnMicrowave": {
        "microwave_started": "Start microwave.",
        "gripper_released": "Release microwave button.",
    },
    "TurnOnSinkFaucet": {
        "water_on": "Turn on water.",
    },
}


def _plain_label(name):
    return _OBJECT_LABELS.get(name, name.replace("_", " "))


def _describe_relation(name, relation):
    left, right = name.split(relation, 1)
    left = left.rstrip("_")
    right = right.lstrip("_")
    relation_text = relation.strip("_").replace("_", " ")
    if relation_text in {"in", "on", "under"}:
        return f"Move {_plain_label(left)} to {_plain_label(right)}."
    if relation_text == "next to":
        return f"Move {_plain_label(left)} next to {_plain_label(right)}."
    return f"Move {_plain_label(left)} to {_plain_label(right)}."


def _describe_predicate(name, predicate):
    if description := predicate.get("description"):
        return description
    if name in _PREDICATE_DESCRIPTION_OVERRIDES:
        return _PREDICATE_DESCRIPTION_OVERRIDES[name]

    if name.endswith("_grasped"):
        return f"Pick {_plain_label(name.removesuffix('_grasped'))}."
    if name.endswith("_released"):
        return f"Release {_plain_label(name.removesuffix('_released'))}."
    if name.endswith("_open"):
        return f"Open {_plain_label(name.removesuffix('_open'))}."
    if name.endswith("_closed"):
        return f"Close {_plain_label(name.removesuffix('_closed'))}."
    if name.endswith("_started"):
        return f"Start {_plain_label(name.removesuffix('_started'))}."
    if "_in_" in name:
        return _describe_relation(name, "_in_")
    if "_on_" in name:
        return _describe_relation(name, "_on_")
    if "_under_" in name:
        return _describe_relation(name, "_under_")
    if "_next_to_" in name:
        return _describe_relation(name, "_next_to_")
    if name.endswith("_upright"):
        return f"Keep {_plain_label(name.removesuffix('_upright'))} upright."

    text = name.replace("_", " ")
    return f"Complete the subtask condition: {text}."


def _with_descriptions(predicates, task_name=None):
    task_descriptions = _TASK_PREDICATE_DESCRIPTION_OVERRIDES.get(task_name, {})
    described = {}
    for name, predicate in predicates.items():
        entry = dict(predicate)
        entry.setdefault(
            "description",
            task_descriptions.get(name) or _describe_predicate(name, entry),
        )
        described[name] = entry
    return described


def _safe(default, fn):
    try:
        return fn()
    except Exception:
        return default


def _p(value, required=None, stage="object_state", source="eval_composite_predicate"):
    if required is None:
        required = stage != "transient"
    return {
        "value": bool(value),
        "required": required,
        "stage": stage,
        "source": source,
    }


def _OU():
    import robocasa.utils.object_utils as OU

    return OU


def _in(env, obj_name, receptacle_name, **kwargs):
    return _safe(
        False,
        lambda: _OU().check_obj_in_receptacle(env, obj_name, receptacle_name, **kwargs),
    )


def _inside(env, obj_name, fixture, **kwargs):
    return _safe(False, lambda: _OU().obj_inside_of(env, obj_name, fixture, **kwargs))


def _on_fixture(env, obj_name, fixture):
    return _safe(False, lambda: _OU().check_obj_fixture_contact(env, obj_name, fixture))


def _far(env, *obj_names, **kwargs):
    return all(
        _safe(False, lambda obj=obj: _OU().gripper_obj_far(env, obj, **kwargs))
        for obj in obj_names
    )


def _grasped(env, obj_name):
    return _safe(False, lambda: _OU().check_obj_grasped(env, obj_name))


def _is_open(env, fixture):
    return _safe(False, lambda: fixture.is_open(env))


def _is_closed(env, fixture, **kwargs):
    return _safe(
        False,
        lambda: fixture.is_closed(env=env, **kwargs),
    )


def _sink_water_on(env, sink):
    return _safe(False, lambda: sink.get_handle_state(env=env)["water_on"])


def _obj_xy(env, obj_name):
    return np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]])[:2]


def _xy_dist(env, obj_name1, obj_name2):
    return float(np.linalg.norm(_obj_xy(env, obj_name1) - _obj_xy(env, obj_name2)))


def _stove_burner_on(env, stove, burner_name):
    return _safe(False, lambda: stove.is_burner_on(env=env, burner_loc=burner_name))


def _stove_obj_on_burner(env, stove, obj_name, burner_name=None, threshold=0.15):
    obj_pos = _safe(None, lambda: _obj_xy(env, obj_name))
    if obj_pos is None or not _on_fixture(env, obj_name, stove):
        return False

    for location, site in stove.burner_sites.items():
        if burner_name is not None and location != burner_name:
            continue
        if site is None:
            continue
        burner_pos = np.array(env.sim.data.get_site_xpos(site.get("name")))[:2]
        if np.linalg.norm(burner_pos - obj_pos) < threshold:
            return True
    return False


def _toaster_any_slot_on(env):
    def check():
        state = env.toaster.get_state(env)
        for slot_pair in range(len(state.keys())):
            if env.toaster.check_slot_contact(env, "obj", slot_pair=slot_pair):
                return bool(
                    env.toaster.get_state(env, slot_pair=slot_pair)["turned_on"]
                )
        return bool(getattr(env, "toaster_on", False))

    return _safe(False, check)


def _toaster_oven_time(env, fixture):
    return _safe(0.0, lambda: fixture.get_state(env)["time"])


def _toaster_oven_closed(env, fixture):
    return _safe(False, lambda: not fixture.is_open(env))


def _microwave_on(env, fixture):
    return _safe(False, lambda: fixture.get_state()["turned_on"])


def _dishwasher_rack_contact(env, obj_name):
    return _safe(False, lambda: env.dishwasher.check_rack_contact(env, obj_name))


def _fridge_rack_contact(env, obj_name, **kwargs):
    return _safe(False, lambda: env.fridge.check_rack_contact(env, obj_name, **kwargs))


def _deliver_straw(env):
    straw_in_glass = _in(env, "straw", "glass_cup", th=0.5)
    return {
        "drawer_open": _p(_is_open(env, env.drawer), stage="precondition"),
        "straw_grasped": _p(_grasped(env, "straw"), stage="transient"),
        "straw_in_glass": _p(straw_in_glass, stage="placement"),
        "gripper_released": _p(_far(env, "straw"), stage="release"),
    }


def _get_toasted_bread(env):
    return {
        "toaster_started": _p(_toaster_any_slot_on(env), stage="control"),
        "bread_grasped": _p(_grasped(env, "obj"), stage="transient"),
        "bread_on_plate": _p(_in(env, "obj", "plate"), stage="placement"),
        "gripper_released": _p(_far(env, "obj"), stage="release"),
    }


def _kettle_boiling(env):
    kettle_on_burner = _stove_obj_on_burner(env, env.stove, "obj")
    burner_on = any(
        _stove_burner_on(env, env.stove, location)
        for location in env.stove.get_knobs_state(env=env).keys()
    )
    return {
        "kettle_grasped": _p(_grasped(env, "obj"), stage="transient"),
        "kettle_on_burner": _p(kettle_on_burner, stage="placement"),
        "burner_on": _p(burner_on, stage="control"),
        "gripper_released": _p(_far(env, "obj"), stage="release"),
    }


def _load_dishwasher(env):
    dishes_on_rack = all(
        _dishwasher_rack_contact(env, name) for name in ["dish0", "dish1"]
    )
    return {
        "dishwasher_rack_accessible": _p(
            _is_open(env, env.dishwasher), stage="diagnostic"
        ),
        "dishes_grasped": _p(
            _grasped(env, "dish0") or _grasped(env, "dish1"),
            stage="transient",
        ),
        "dishes_on_rack": _p(dishes_on_rack, stage="placement"),
        "dishwasher_closed": _p(
            _is_closed(env, env.dishwasher, th=0.05), stage="fixture_state"
        ),
    }


def _make_ice_lemonade(env):
    ice_in_glass = _in(env, "ice_cube1", "glass_cup", th=0.5) or _in(
        env, "ice_cube2", "glass_cup", th=0.5
    )
    return {
        "fridge_open": _p(
            _is_open(env, env.fridge),
            stage="precondition",
            source="fixture_state",
        ),
        "lemon_grasped": _p(
            _grasped(env, "lemon_wedge"),
            stage="transient",
            source="diagnostic",
        ),
        "ice_cube1_grasped": _p(
            _grasped(env, "ice_cube1"),
            stage="transient",
            source="diagnostic",
        ),
        "ice_cube2_grasped": _p(
            _grasped(env, "ice_cube2"),
            stage="transient",
            source="diagnostic",
        ),
        "lemon_in_glass": _p(
            _in(env, "lemon_wedge", "glass_cup", th=0.5),
            stage="placement",
            source="_check_success",
        ),
        "ice_in_glass": _p(
            ice_in_glass,
            stage="placement",
            source="_check_success",
        ),
        "gripper_released": _p(
            _far(env, "lemon_wedge", "ice_cube1", "ice_cube2", th=0.15),
            stage="release",
            source="_check_success",
        ),
    }


def _pack_identical_lunches(env):
    veg_in_0 = [
        veg for veg in ["vegetable0", "vegetable1"] if _in(env, veg, "tupperware0")
    ]
    veg_in_1 = [
        veg for veg in ["vegetable0", "vegetable1"] if _in(env, veg, "tupperware1")
    ]
    meat_in_0 = [meat for meat in ["meat0", "meat1"] if _in(env, meat, "tupperware0")]
    meat_in_1 = [meat for meat in ["meat0", "meat1"] if _in(env, meat, "tupperware1")]
    all_objs = veg_in_0 + veg_in_1 + meat_in_0 + meat_in_1
    no_duplicates = len(all_objs) == len(set(all_objs))
    return {
        "fridge_open": _p(_is_open(env, env.fridge), stage="precondition"),
        "tupperware0_has_one_vegetable": _p(len(veg_in_0) == 1, stage="placement"),
        "tupperware0_has_one_meat": _p(len(meat_in_0) == 1, stage="placement"),
        "tupperware1_has_one_vegetable": _p(len(veg_in_1) == 1, stage="placement"),
        "tupperware1_has_one_meat": _p(len(meat_in_1) == 1, stage="placement"),
        "objects_not_duplicated": _p(
            no_duplicates and len(all_objs) == 4, stage="object_state"
        ),
        "gripper_released": _p(
            _far(env, *all_objs) if all_objs else False, stage="release"
        ),
    }


def _pre_soak_pan(env):
    return {
        "pan_grasped": _p(_grasped(env, "obj1"), stage="transient"),
        "pan_in_sink": _p(
            _inside(env, "obj1", env.sink, partial_check=False), stage="placement"
        ),
        "sponge_grasped": _p(_grasped(env, "obj2"), stage="transient"),
        "sponge_in_sink": _p(
            _inside(env, "obj2", env.sink, partial_check=False), stage="placement"
        ),
        "water_on": _p(_sink_water_on(env, env.sink), stage="control"),
        "gripper_released": _p(_far(env, "obj1", "obj2"), stage="release"),
    }


def _prepare_coffee(env):
    return {
        "mug_grasped": _p(_grasped(env, "obj"), stage="transient"),
        "mug_under_dispenser": _p(
            _safe(
                False,
                lambda: env.coffee_machine.check_receptacle_placement_for_pouring(
                    env, "obj"
                ),
            ),
            stage="placement",
        ),
        "coffee_started": _p(
            _safe(False, lambda: env.coffee_machine._turned_on), stage="control"
        ),
        "gripper_released": _p(
            _far(env, "obj")
            and _safe(False, lambda: env.coffee_machine.gripper_button_far(env)),
            stage="release",
        ),
    }


def _rinse_sink_basin(env):
    handle_state = _safe({}, lambda: env.sink.get_handle_state(env=env))
    return {
        "water_on": _p(bool(handle_state.get("water_on", False)), stage="control"),
        "left_basin_rinsed": _p(
            bool(getattr(env, "washed_loc", [False, False, False])[0]), stage="temporal"
        ),
        "center_basin_rinsed": _p(
            bool(getattr(env, "washed_loc", [False, False, False])[1]), stage="temporal"
        ),
        "right_basin_rinsed": _p(
            bool(getattr(env, "washed_loc", [False, False, False])[2]), stage="temporal"
        ),
    }


def _scrub_cutting_board(env):
    sweep_range = 0.0
    positions = getattr(env, "board_contact_positions", [])
    if positions:
        positions = np.array(positions)
        sweep_range = float(
            np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
        )
    return {
        "sponge_grasped": _p(_grasped(env, "sponge"), stage="transient"),
        "board_contact_count_reached": _p(
            getattr(env, "board_contact_timer", 0) >= 5, stage="temporal"
        ),
        "board_sweep_range_reached": _p(sweep_range >= 0.1, stage="temporal"),
        "gripper_released": _p(_far(env, "sponge", th=0.15), stage="release"),
    }


def _searing_meat(env):
    pan_on_burner = _stove_obj_on_burner(env, env.stove, "pan", burner_name=env.knob)
    return {
        "cabinet_open": _p(_is_open(env, env.cab), stage="precondition"),
        "pan_grasped": _p(_grasped(env, "pan"), stage="transient"),
        "pan_on_target_burner": _p(pan_on_burner, stage="placement"),
        "meat_grasped": _p(_grasped(env, "meat"), stage="transient"),
        "meat_in_pan": _p(_in(env, "meat", "pan", th=0.07), stage="placement"),
        "burner_on": _p(_stove_burner_on(env, env.stove, env.knob), stage="control"),
        "gripper_released": _p(_far(env, "meat"), stage="release"),
    }


def _set_up_cutting_station(env):
    return {
        "drawer_open": _p(_is_open(env, env.drawer), stage="precondition"),
        "knife_grasped": _p(_grasped(env, "knife"), stage="transient"),
        "knife_on_cutting_board": _p(
            _in(env, "knife", "receptacle"), stage="placement"
        ),
        "meat_grasped": _p(_grasped(env, "meat"), stage="transient"),
        "meat_on_cutting_board": _p(_in(env, "meat", "receptacle"), stage="placement"),
        "gripper_released": _p(_far(env, "knife", "receptacle"), stage="release"),
    }


def _stack_bowls_cabinet(env):
    bowl1_in_cabinet = _inside(env, "bowl1", env.cabinet)
    bowl2_in_cabinet = _inside(env, "bowl2", env.cabinet)
    stacked = _in(env, "bowl2", "bowl1") or _in(env, "bowl1", "bowl2")
    return {
        "cabinet_open": _p(_is_open(env, env.cabinet), stage="precondition"),
        "larger_bowl_in_cabinet": _p(bowl2_in_cabinet, stage="placement"),
        "smaller_bowl_in_cabinet": _p(bowl1_in_cabinet, stage="placement"),
        "bowls_stacked": _p(stacked, stage="placement"),
        "gripper_released": _p(_far(env, "bowl1", "bowl2"), stage="release"),
    }


def _steam_in_microwave(env):
    return {
        "vegetable_in_bowl": _p(_in(env, "vegetable", "bowl"), stage="placement"),
        "bowl_in_microwave": _p(_inside(env, "bowl", env.microwave), stage="placement"),
        "microwave_closed": _p(_is_closed(env, env.microwave), stage="fixture_state"),
        "microwave_started": _p(_microwave_on(env, env.microwave), stage="control"),
    }


def _stir_vegetables(env):
    return {
        "vegetable1_in_pot": _p(_in(env, "veg1", "pot"), stage="placement"),
        "vegetable2_in_pot": _p(_in(env, "veg2", "pot"), stage="placement"),
        "spatula_grasped": _p(_grasped(env, "spatula"), stage="transient"),
        "vegetables_stirred": _p(
            getattr(env, "success_time", 0) >= 5, stage="temporal"
        ),
        "spatula_released": _p(_far(env, "spatula"), stage="release"),
    }


def _store_leftovers_in_bowl(env):
    return {
        "chicken_in_bowl": _p(_in(env, "chicken_drumstick", "bowl"), stage="placement"),
        "vegetable_in_bowl": _p(_in(env, "vegetable", "bowl"), stage="placement"),
        "bowl_in_fridge": _p(_fridge_rack_contact(env, "bowl"), stage="placement"),
        "gripper_released": _p(_far(env, "bowl"), stage="release"),
    }


def _wash_lettuce(env):
    return {
        "water_on": _p(_sink_water_on(env, env.sink), stage="control"),
        "lettuce_under_water": _p(
            _safe(False, lambda: env.sink.check_obj_under_water(env, "lettuce")),
            stage="placement",
        ),
        "washed_time_reached": _p(
            getattr(env, "washed_time", 0) >= 25, stage="temporal"
        ),
    }


def _arrange_bread_basket(env):
    return {
        "cabinet_open": _p(_is_open(env, env.cab), stage="precondition"),
        "bread_in_basket": _p(_in(env, "bread", "basket"), stage="placement"),
        "basket_on_dining_counter": _p(
            _on_fixture(env, "basket", env.dining_table), stage="placement"
        ),
        "gripper_released": _p(_far(env, "basket"), stage="release"),
    }


def _arrange_tea(env):
    return {
        "kettle_on_tray": _p(_in(env, "obj2", "container"), stage="placement"),
        "mug_on_tray": _p(_in(env, "obj", "container"), stage="placement"),
        "cabinet_closed": _p(_is_closed(env, env.cab), stage="fixture_state"),
        "gripper_released": _p(_far(env, "obj"), stage="release"),
    }


def _bread_selection(env):
    return {
        "croissant_on_cutting_board": _p(
            _in(env, "croissant", "cutting_board"), stage="placement"
        ),
        "jam_on_cutting_board": _p(_in(env, "jam", "cutting_board"), stage="placement"),
        "gripper_released": _p(
            _far(env, "croissant") and _far(env, "jam"), stage="release"
        ),
    }


def _categorize_condiments(env):
    return {
        "bottle_in_cabinet": _p(_inside(env, "obj1", env.cab), stage="placement"),
        "shaker_in_cabinet": _p(_inside(env, "obj2", env.cab), stage="placement"),
        "bottle_next_to_counterpart": _p(
            _safe(False, lambda: env._xy_dist("obj1", "cab_obj1") <= 0.15),
            stage="placement",
        ),
        "shaker_next_to_counterpart": _p(
            _safe(False, lambda: env._xy_dist("obj2", "cab_obj2") <= 0.15),
            stage="placement",
        ),
        "gripper_released": _p(_far(env, "obj1", "obj2"), stage="release"),
    }


def _cutting_tool_selection(env):
    peeler_on_board = _in(env, "peeler", "food_container")
    knife_on_board = _in(env, "knife", "food_container")
    correct_tool = env._CUTTING_MAP[env.food]
    correct_tool_chosen = (
        knife_on_board and not peeler_on_board
        if correct_tool == "knife"
        else peeler_on_board and not knife_on_board
    )
    return {
        "drawer_open": _p(_is_open(env, env.drawer), stage="precondition"),
        "correct_tool_grasped": _p(_grasped(env, correct_tool), stage="transient"),
        "correct_tool_on_cutting_board": _p(correct_tool_chosen, stage="placement"),
        "wrong_tool_not_on_cutting_board": _p(
            (not peeler_on_board) if correct_tool == "knife" else (not knife_on_board),
            stage="placement",
        ),
        "gripper_released": _p(_far(env, "food_container"), stage="release"),
    }


def _garnish_pancake(env):
    return {
        "fridge_open": _p(_is_open(env, env.fridge), stage="precondition"),
        "strawberry_grasped": _p(_grasped(env, "strawberry"), stage="transient"),
        "pancake_on_plate": _p(
            _in(env, "pancake", "pancake_container"), stage="placement"
        ),
        "plate_on_table": _p(
            _on_fixture(env, "pancake_container", env.dining_counter), stage="placement"
        ),
        "strawberry_on_pancake": _p(
            _in(env, "strawberry", "pancake"), stage="placement"
        ),
        "gripper_released": _p(_far(env, "strawberry"), stage="release"),
    }


def _gather_tableware(env):
    glass1_glass2 = _xy_dist(env, "glass1", "glass2")
    glass2_glass3 = _xy_dist(env, "glass2", "glass3")
    glass3_glass1 = _xy_dist(env, "glass3", "glass1")
    bowl_glass1 = _xy_dist(env, "bowl", "glass1")
    bowl_glass2 = _xy_dist(env, "bowl", "glass2")
    bowl_glass3 = _xy_dist(env, "bowl", "glass3")
    glasses_clustered = max(glass1_glass2, glass2_glass3, glass3_glass1) * 1.5 < max(
        bowl_glass1, bowl_glass2, bowl_glass3
    )
    return {
        "cabinets_open": _p(
            _is_open(env, env.cab) and _is_open(env, env.cab2),
            stage="precondition",
        ),
        "glasses_clustered": _p(glasses_clustered, stage="placement"),
        "bowl_separated_from_glasses": _p(glasses_clustered, stage="placement"),
        "gripper_released": _p(
            _far(env, "glass1", "glass2", "glass3"), stage="release"
        ),
    }


def _heat_kebab_sandwich(env):
    baguette_in_toaster = _safe(
        False,
        lambda: env.toaster_oven.check_rack_contact(env, "baguette", rack_level=0)
        or env.toaster_oven.check_rack_contact(env, "baguette", rack_level=1),
    )
    kebab_in_toaster = _safe(
        False,
        lambda: env.toaster_oven.check_rack_contact(env, "kebab", rack_level=0)
        or env.toaster_oven.check_rack_contact(env, "kebab", rack_level=1),
    )
    return {
        "kebab_in_toaster_oven": _p(kebab_in_toaster, stage="placement"),
        "baguette_in_toaster_oven": _p(baguette_in_toaster, stage="placement"),
        "toaster_oven_closed": _p(
            _toaster_oven_closed(env, env.toaster_oven), stage="fixture_state"
        ),
        "timer_set": _p(
            _toaster_oven_time(env, env.toaster_oven) >= 0.1, stage="control"
        ),
    }


def _pan_transfer(env):
    return {
        "pan_grasped": _p(_grasped(env, "vegetable_container"), stage="transient"),
        "vegetable_on_plate": _p(_in(env, "vegetable", "plate"), stage="placement"),
        "pan_on_stove": _p(
            _safe(
                False,
                lambda: env._check_obj_location_on_stove("vegetable_container")
                is not None,
            ),
            stage="placement",
        ),
        "robot_did_not_touch_food": _p(
            not getattr(env, "_robot_touched_food", False), stage="constraint"
        ),
        "gripper_released": _p(_far(env, "vegetable_container"), stage="release"),
    }


def _portion_hot_dogs(env):
    buns = ["hotdog_bun1", "hotdog_bun2"]
    sausages = ["sausage1", "sausage2"]
    buns_in_plate1 = sum(_in(env, bun, "plate1") for bun in buns)
    buns_in_plate2 = sum(_in(env, bun, "plate2") for bun in buns)
    sausages_in_plate1 = sum(_in(env, sausage, "plate1") for sausage in sausages)
    sausages_in_plate2 = sum(_in(env, sausage, "plate2") for sausage in sausages)
    return {
        "plate1_has_one_bun": _p(buns_in_plate1 == 1, stage="placement"),
        "plate1_has_one_sausage": _p(sausages_in_plate1 == 1, stage="placement"),
        "plate2_has_one_bun": _p(buns_in_plate2 == 1, stage="placement"),
        "plate2_has_one_sausage": _p(sausages_in_plate2 == 1, stage="placement"),
        "gripper_released": _p(_far(env, *(buns + sausages)), stage="release"),
    }


def _recycle_bottles_by_type(env):
    plastic = ["bottle_plastic1", "bottle_plastic2", "bottle_plastic_middle"]
    glass = ["bottle_glass1", "bottle_glass2", "bottle_glass_middle"]
    if getattr(env, "choice", None) == "alcohol":
        glass.append("mystery_middle")
    else:
        plastic.append("mystery_middle")

    def cluster_okay(names, thresh=0.30):
        sample_names = [name for name in names if not name.endswith("_middle")]
        middle_names = [name for name in names if name.endswith("_middle")]
        for middle_name in middle_names:
            dists = [
                _xy_dist(env, middle_name, sample_name) for sample_name in sample_names
            ]
            if not any(dist <= thresh for dist in dists):
                return False
        return True

    all_names = plastic + glass
    return {
        "plastic_bottles_clustered": _p(cluster_okay(plastic), stage="placement"),
        "glass_bottles_clustered": _p(cluster_okay(glass), stage="placement"),
        "bottles_on_table": _p(
            all(_on_fixture(env, name, env.dining_counter) for name in all_names),
            stage="placement",
        ),
        "gripper_released": _p(_far(env, *all_names), stage="release"),
    }


def _separate_freezer_rack(env):
    return {
        "freezer_open": _p(_is_open(env, env.fridge), stage="precondition"),
        "meat_in_tupperware": _p(
            _in(env, "meat", "meat_tupperware"), stage="object_state"
        ),
        "vegetables_in_tupperware": _p(
            _in(env, "vegetable1", "veg_tupperware")
            and _in(env, "vegetable2", "veg_tupperware"),
            stage="object_state",
        ),
        "meat_container_on_second_rack": _p(
            _fridge_rack_contact(
                env, "meat_tupperware", compartment="freezer", rack_index=-2
            ),
            stage="placement",
        ),
        "vegetable_container_on_top_rack": _p(
            _fridge_rack_contact(
                env, "veg_tupperware", compartment="freezer", rack_index=-1
            ),
            stage="placement",
        ),
        "gripper_released": _p(
            _far(env, "meat_tupperware", "veg_tupperware"), stage="release"
        ),
    }


def _waffle_reheat(env):
    return {
        "microwave_open": _p(_is_open(env, env.microwave), stage="precondition"),
        "waffle_in_bowl": _p(
            _in(env, "waffle", "waffle_container"), stage="object_state"
        ),
        "bowl_in_microwave": _p(
            _inside(env, "waffle_container", env.microwave), stage="placement"
        ),
        "microwave_closed": _p(_is_closed(env, env.microwave), stage="fixture_state"),
        "microwave_started": _p(_microwave_on(env, env.microwave), stage="control"),
    }


def _wash_fruit_colander(env):
    fruit_in_colander = all(
        _in(env, f"fruit{i}", "colander") for i in range(env.num_fruit)
    )
    return {
        "colander_in_sink": _p(_inside(env, "colander", env.sink), stage="placement"),
        "fruit_in_colander": _p(fruit_in_colander, stage="placement"),
        "colander_under_water": _p(
            _safe(False, lambda: env.sink.check_obj_under_water(env, "colander")),
            stage="control",
        ),
    }


def _weigh_ingredients(env):
    return {
        "packaged_food_grasped": _p(_grasped(env, "obj"), stage="transient"),
        "packaged_food_on_scale": _p(
            _in(env, "obj", "digital_scale"), stage="placement"
        ),
        "packaged_food_upright": _p(
            _safe(False, lambda: _OU().check_obj_upright(env, "obj")),
            stage="object_state",
        ),
        "cabinet_closed": _p(_is_closed(env, env.cab), stage="fixture_state"),
        "gripper_released": _p(_far(env, "obj"), stage="release"),
    }


def _close_blender_lid(env):
    lid_body = _safe(None, lambda: f"{env.blender.blender_lid.name}_main")
    return {
        "blender_lid_on_blender": _p(
            _safe(False, lambda: env.blender.get_state()["lid_on_blender"]),
            stage="placement",
        ),
        "gripper_released": _p(
            _safe(False, lambda: _OU().gripper_fxtr_far(env, lid_body, th=0.15))
            if lid_body
            else False,
            stage="release",
        ),
    }


def _close_fridge(env):
    return {"fridge_closed": _p(_is_closed(env, env.fxtr), stage="fixture_state")}


def _close_toaster_oven_door(env):
    return {
        "toaster_oven_closed": _p(
            _safe(False, lambda: env.toaster_oven.is_closed(env)),
            stage="fixture_state",
        )
    }


def _coffee_setup_mug(env):
    return {
        "mug_grasped": _p(_grasped(env, "obj"), stage="transient"),
        "mug_under_dispenser": _p(
            _safe(
                False,
                lambda: env.coffee_machine.check_receptacle_placement_for_pouring(
                    env, "obj"
                ),
            ),
            stage="placement",
        ),
        "gripper_released": _p(_far(env, "obj"), stage="release"),
    }


def _navigate_kitchen(env):
    def base_orientation():
        import robosuite.utils.transform_utils as T

        return T.mat2euler(np.array(env.sim.data.body_xmat[robot_id]).reshape((3, 3)))

    robot_id = _safe(None, lambda: env.sim.model.body_name2id("mobilebase0_base"))
    base_pos = _safe(None, lambda: np.array(env.sim.data.body_xpos[robot_id]))
    base_ori = _safe(None, base_orientation)
    at_target = (
        bool(np.linalg.norm(env.target_pos[:2] - base_pos[:2]) <= 0.20)
        if base_pos is not None
        else False
    )
    facing_target = (
        bool(np.cos(env.target_ori[2] - base_ori[2]) >= 0.98)
        if base_ori is not None
        else False
    )
    return {
        "at_target_position": _p(at_target, stage="navigation"),
        "facing_target": _p(facing_target, stage="navigation"),
    }


def _open_cabinet(env):
    return {"cabinet_open": _p(_is_open(env, env.fxtr), stage="fixture_state")}


def _open_drawer(env):
    door_state = _safe({}, lambda: env.drawer.get_door_state(env=env))
    return {
        "drawer_open": _p(
            bool(door_state)
            and all(joint_p >= 0.95 for joint_p in door_state.values()),
            stage="fixture_state",
        )
    }


def _open_stand_mixer_head(env):
    return {
        "stand_mixer_head_open": _p(
            _safe(False, lambda: env.stand_mixer.get_state(env)["head"] > 0.99),
            stage="fixture_state",
        )
    }


def _pick_place_target(env, target_predicates):
    predicates = {"object_grasped": _p(_grasped(env, "obj"), stage="transient")}
    predicates.update(target_predicates)
    predicates["gripper_released"] = _p(_far(env, "obj"), stage="release")
    predicates["final_placement_valid"] = _p(
        env._check_success(), stage="task_success"
    )
    return predicates


def _pick_place_counter_to_cabinet(env):
    return _pick_place_target(
        env,
        {
            "object_in_cabinet": _p(
                _inside(env, "obj", env.cab),
                stage="placement",
            )
        },
    )


def _pick_place_counter_to_stove(env):
    return _pick_place_target(
        env,
        {
            "object_in_pan": _p(
                _in(env, "obj", "container", th=0.07),
                stage="placement",
            )
        },
    )


def _pick_place_drawer_to_counter(env):
    return _pick_place_target(
        env,
        {
            "object_on_counter": _p(
                _safe(False, lambda: _OU().check_obj_any_counter_contact(env, "obj")),
                stage="placement",
            )
        },
    )


def _pick_place_sink_to_counter(env):
    return _pick_place_target(
        env,
        {
            "object_in_container": _p(
                _in(env, "obj", "container"),
                stage="placement",
            ),
            "container_on_counter": _p(
                _safe(
                    False,
                    lambda: env.check_contact(env.objects["container"], env.counter),
                ),
                stage="placement",
            ),
        },
    )


def _pick_place_toaster_to_counter(env):
    return _pick_place_target(
        env,
        {
            "object_on_plate": _p(
                _in(env, "obj", "plate"),
                stage="placement",
            )
        },
    )


def _slide_dishwasher_rack(env):
    current_pos = _safe(None, lambda: env.dishwasher.get_state(env)["rack"])
    if current_pos is None:
        rack_slid = False
    elif getattr(env, "should_pull", False):
        rack_slid = current_pos >= 0.95
    else:
        rack_slid = current_pos <= 0.05
    return {"dishwasher_rack_slid": _p(rack_slid, stage="fixture_state")}


def _turn_off_stove(env):
    return {
        "burner_off": _p(
            not _stove_burner_on(env, env.stove, env.knob),
            stage="control",
        )
    }


def _turn_on_electric_kettle(env):
    return {
        "electric_kettle_on": _p(
            _safe(False, lambda: env.electric_kettle.get_state(env)["turned_on"]),
            stage="control",
        )
    }


def _turn_on_microwave(env):
    return {
        "microwave_started": _p(_microwave_on(env, env.microwave), stage="control"),
        "gripper_released": _p(
            _safe(
                False,
                lambda: env.microwave.gripper_button_far(env, button="start_button"),
            ),
            stage="release",
        ),
    }


def _turn_on_sink_faucet(env):
    handle_state = _safe({}, lambda: env.sink.get_handle_state(env=env))
    return {"water_on": _p(bool(handle_state.get("water_on", False)), stage="control")}


def _generic_atomic_success(env):
    return {"official_success": _p(env._check_success(), stage="task_success")}


def _generic_pick_place_success(env):
    return {
        "object_grasped": _p(_grasped(env, "obj"), stage="transient"),
        "object_at_target_and_released": _p(env._check_success(), stage="task_success"),
    }


EVAL_COMPOSITE_PREDICATES = {
    "DeliverStraw": _deliver_straw,
    "GetToastedBread": _get_toasted_bread,
    "KettleBoiling": _kettle_boiling,
    "LoadDishwasher": _load_dishwasher,
    "MakeIceLemonade": _make_ice_lemonade,
    "PackIdenticalLunches": _pack_identical_lunches,
    "PreSoakPan": _pre_soak_pan,
    "PrepareCoffee": _prepare_coffee,
    "RinseSinkBasin": _rinse_sink_basin,
    "ScrubCuttingBoard": _scrub_cutting_board,
    "SearingMeat": _searing_meat,
    "SetUpCuttingStation": _set_up_cutting_station,
    "StackBowlsCabinet": _stack_bowls_cabinet,
    "SteamInMicrowave": _steam_in_microwave,
    "StirVegetables": _stir_vegetables,
    "StoreLeftoversInBowl": _store_leftovers_in_bowl,
    "WashLettuce": _wash_lettuce,
    "ArrangeBreadBasket": _arrange_bread_basket,
    "ArrangeTea": _arrange_tea,
    "BreadSelection": _bread_selection,
    "CategorizeCondiments": _categorize_condiments,
    "CuttingToolSelection": _cutting_tool_selection,
    "GarnishPancake": _garnish_pancake,
    "GatherTableware": _gather_tableware,
    "HeatKebabSandwich": _heat_kebab_sandwich,
    "PanTransfer": _pan_transfer,
    "PortionHotDogs": _portion_hot_dogs,
    "RecycleBottlesByType": _recycle_bottles_by_type,
    "SeparateFreezerRack": _separate_freezer_rack,
    "WaffleReheat": _waffle_reheat,
    "WashFruitColander": _wash_fruit_colander,
    "WeighIngredients": _weigh_ingredients,
}


ATOMIC_TASK_PREDICATES = {
    "CloseBlenderLid": _close_blender_lid,
    "CloseFridge": _close_fridge,
    "CloseToasterOvenDoor": _close_toaster_oven_door,
    "CoffeeSetupMug": _coffee_setup_mug,
    "NavigateKitchen": _navigate_kitchen,
    "OpenCabinet": _open_cabinet,
    "OpenDrawer": _open_drawer,
    "OpenStandMixerHead": _open_stand_mixer_head,
    "PickPlaceCounterToCabinet": _pick_place_counter_to_cabinet,
    "PickPlaceCounterToStove": _pick_place_counter_to_stove,
    "PickPlaceDrawerToCounter": _pick_place_drawer_to_counter,
    "PickPlaceSinkToCounter": _pick_place_sink_to_counter,
    "PickPlaceToasterToCounter": _pick_place_toaster_to_counter,
    "SlideDishwasherRack": _slide_dishwasher_rack,
    "TurnOffStove": _turn_off_stove,
    "TurnOnElectricKettle": _turn_on_electric_kettle,
    "TurnOnMicrowave": _turn_on_microwave,
    "TurnOnSinkFaucet": _turn_on_sink_faucet,
}


def get_eval_composite_subtask_predicates(env):
    task_name = env.__class__.__name__
    fn = EVAL_COMPOSITE_PREDICATES.get(task_name)
    if fn is None:
        return {}
    return _with_descriptions(fn(env), task_name=task_name)


def get_atomic_subtask_predicates(env):
    task_name = env.__class__.__name__
    fn = ATOMIC_TASK_PREDICATES.get(task_name)
    if fn is None:
        if task_name.startswith("PickPlace"):
            return _with_descriptions(
                _generic_pick_place_success(env), task_name=task_name
            )
        return _with_descriptions(_generic_atomic_success(env), task_name=task_name)
    return _with_descriptions(fn(env), task_name=task_name)


def get_runtime_subtask_predicates(env):
    task_name = env.__class__.__name__
    if task_name in EVAL_COMPOSITE_PREDICATES:
        return get_eval_composite_subtask_predicates(env)
    return get_atomic_subtask_predicates(env)
