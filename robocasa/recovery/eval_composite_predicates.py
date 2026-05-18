"""Runtime subtask predicates for the 32 target composite eval tasks.

These predicates are a monitoring layer only. They decompose task progress for
debugging and recovery, but they do not replace each task's official
``_check_success()`` evaluator.
"""

from __future__ import annotations

import numpy as np


def _safe(default, fn):
    try:
        return fn()
    except Exception:
        return default


def _p(value, required=True, stage="object_state", source="eval_composite_predicate"):
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
        "drawer_open": _p(
            _is_open(env, env.drawer), required=False, stage="precondition"
        ),
        "straw_grasped": _p(_grasped(env, "straw"), required=False, stage="transient"),
        "straw_in_glass": _p(straw_in_glass, stage="placement"),
        "gripper_released": _p(_far(env, "straw"), stage="release"),
    }


def _get_toasted_bread(env):
    return {
        "toaster_started": _p(_toaster_any_slot_on(env), stage="control"),
        "bread_grasped": _p(_grasped(env, "obj"), required=False, stage="transient"),
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
        "kettle_grasped": _p(_grasped(env, "obj"), required=False, stage="transient"),
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
            _is_open(env, env.dishwasher), required=False, stage="diagnostic"
        ),
        "dishes_grasped": _p(
            _grasped(env, "dish0") or _grasped(env, "dish1"),
            required=False,
            stage="transient",
        ),
        "dishes_on_rack": _p(dishes_on_rack, stage="placement"),
        "dishwasher_closed": _p(
            _is_closed(env, env.dishwasher, th=0.05), stage="fixture_state"
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
        "fridge_open": _p(
            _is_open(env, env.fridge), required=False, stage="precondition"
        ),
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
        "pan_grasped": _p(_grasped(env, "obj1"), required=False, stage="transient"),
        "pan_in_sink": _p(
            _inside(env, "obj1", env.sink, partial_check=False), stage="placement"
        ),
        "sponge_grasped": _p(_grasped(env, "obj2"), required=False, stage="transient"),
        "sponge_in_sink": _p(
            _inside(env, "obj2", env.sink, partial_check=False), stage="placement"
        ),
        "water_on": _p(_sink_water_on(env, env.sink), stage="control"),
        "gripper_released": _p(_far(env, "obj1", "obj2"), stage="release"),
    }


def _prepare_coffee(env):
    return {
        "mug_grasped": _p(_grasped(env, "obj"), required=False, stage="transient"),
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
        "sponge_grasped": _p(
            _grasped(env, "sponge"), required=False, stage="transient"
        ),
        "board_contact_count_reached": _p(
            getattr(env, "board_contact_timer", 0) >= 5, stage="temporal"
        ),
        "board_sweep_range_reached": _p(sweep_range >= 0.1, stage="temporal"),
        "gripper_released": _p(_far(env, "sponge", th=0.15), stage="release"),
    }


def _searing_meat(env):
    pan_on_burner = _stove_obj_on_burner(env, env.stove, "pan", burner_name=env.knob)
    return {
        "cabinet_open": _p(
            _is_open(env, env.cab), required=False, stage="precondition"
        ),
        "pan_grasped": _p(_grasped(env, "pan"), required=False, stage="transient"),
        "pan_on_target_burner": _p(pan_on_burner, stage="placement"),
        "meat_grasped": _p(_grasped(env, "meat"), required=False, stage="transient"),
        "meat_in_pan": _p(_in(env, "meat", "pan", th=0.07), stage="placement"),
        "burner_on": _p(
            _stove_burner_on(env, env.stove, env.knob), required=False, stage="control"
        ),
        "gripper_released": _p(_far(env, "meat"), stage="release"),
    }


def _set_up_cutting_station(env):
    return {
        "drawer_open": _p(
            _is_open(env, env.drawer), required=False, stage="precondition"
        ),
        "knife_grasped": _p(_grasped(env, "knife"), required=False, stage="transient"),
        "knife_on_cutting_board": _p(
            _in(env, "knife", "receptacle"), stage="placement"
        ),
        "meat_grasped": _p(_grasped(env, "meat"), required=False, stage="transient"),
        "meat_on_cutting_board": _p(_in(env, "meat", "receptacle"), stage="placement"),
        "gripper_released": _p(_far(env, "knife", "receptacle"), stage="release"),
    }


def _stack_bowls_cabinet(env):
    bowl1_in_cabinet = _inside(env, "bowl1", env.cabinet)
    bowl2_in_cabinet = _inside(env, "bowl2", env.cabinet)
    stacked = _in(env, "bowl2", "bowl1") or _in(env, "bowl1", "bowl2")
    return {
        "cabinet_open": _p(
            _is_open(env, env.cabinet), required=False, stage="precondition"
        ),
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
        "spatula_grasped": _p(
            _grasped(env, "spatula"), required=False, stage="transient"
        ),
        "vegetables_stirred": _p(
            getattr(env, "success_time", 0) >= 5, stage="temporal"
        ),
        "spatula_released": _p(_far(env, "spatula"), required=False, stage="release"),
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
        "cabinet_open": _p(
            _is_open(env, env.cab), required=False, stage="precondition"
        ),
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
        "drawer_open": _p(
            _is_open(env, env.drawer), required=False, stage="precondition"
        ),
        "correct_tool_grasped": _p(
            _grasped(env, correct_tool), required=False, stage="transient"
        ),
        "correct_tool_on_cutting_board": _p(correct_tool_chosen, stage="placement"),
        "wrong_tool_not_on_cutting_board": _p(
            (not peeler_on_board) if correct_tool == "knife" else (not knife_on_board),
            stage="placement",
        ),
        "gripper_released": _p(_far(env, "food_container"), stage="release"),
    }


def _garnish_pancake(env):
    return {
        "fridge_open": _p(
            _is_open(env, env.fridge), required=False, stage="precondition"
        ),
        "strawberry_grasped": _p(
            _grasped(env, "strawberry"), required=False, stage="transient"
        ),
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
            required=False,
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
        "pan_grasped": _p(
            _grasped(env, "vegetable_container"), required=False, stage="transient"
        ),
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
        "freezer_open": _p(
            _is_open(env, env.fridge), required=False, stage="precondition"
        ),
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
        "microwave_open": _p(
            _is_open(env, env.microwave), required=False, stage="precondition"
        ),
        "waffle_in_bowl": _p(
            _in(env, "waffle", "waffle_container"), stage="object_state"
        ),
        "bowl_in_microwave": _p(
            _inside(env, "waffle_container", env.microwave), stage="placement"
        ),
        "microwave_closed": _p(
            _is_closed(env, env.microwave), required=False, stage="fixture_state"
        ),
        "microwave_started": _p(_microwave_on(env, env.microwave), stage="control"),
    }


def _wash_fruit_colander(env):
    fruit_in_colander = all(
        _in(env, f"fruit{i}", "colander") for i in range(env.num_fruit)
    )
    return {
        "colander_in_sink": _p(
            _inside(env, "colander", env.sink), required=False, stage="placement"
        ),
        "fruit_in_colander": _p(fruit_in_colander, stage="placement"),
        "colander_under_water": _p(
            _safe(False, lambda: env.sink.check_obj_under_water(env, "colander")),
            stage="control",
        ),
    }


def _weigh_ingredients(env):
    return {
        "packaged_food_grasped": _p(
            _grasped(env, "obj"), required=False, stage="transient"
        ),
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


EVAL_COMPOSITE_PREDICATES = {
    "DeliverStraw": _deliver_straw,
    "GetToastedBread": _get_toasted_bread,
    "KettleBoiling": _kettle_boiling,
    "LoadDishwasher": _load_dishwasher,
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


def get_eval_composite_subtask_predicates(env):
    fn = EVAL_COMPOSITE_PREDICATES.get(env.__class__.__name__)
    if fn is None:
        return {}
    return fn(env)
