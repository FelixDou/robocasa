"""Atomic skill templates used by recovery-oriented task decomposition.

These templates are intentionally policy-agnostic. They describe reusable
monitoring/recovery units that can be inferred from RoboCasa task text and
success predicates, then later connected to concrete controllers.
"""

from __future__ import annotations

from copy import deepcopy


ATOMIC_SKILL_LIBRARY = {
    "open_fixture": {
        "label": "Open fixture",
        "description": "Open an articulated fixture before interacting with contents.",
        "typical_triggers": ["open", "fridge", "freezer", "cabinet", "drawer", "dishwasher", "oven", "microwave"],
        "inputs": ["fixture"],
        "completion_predicates": ["fixture_open", "access_region_visible"],
        "failure_modes": ["fixture_did_not_move", "wrong_fixture_opened", "blocked_handle"],
        "recovery_actions": ["retry_open", "reposition_near_handle", "rewind_to_pre_open_state"],
    },
    "close_fixture": {
        "label": "Close fixture",
        "description": "Close an articulated fixture after object manipulation.",
        "typical_triggers": ["close", "shut", "reset"],
        "inputs": ["fixture"],
        "completion_predicates": ["fixture_closed"],
        "failure_modes": ["fixture_left_open", "blocked_by_object", "wrong_fixture_closed"],
        "recovery_actions": ["retry_close", "clear_blocking_object", "rewind_to_pre_close_state"],
    },
    "pick_object": {
        "label": "Pick object",
        "description": "Acquire an object from its current support or receptacle.",
        "typical_triggers": ["pick", "grab", "take", "retrieve", "gather"],
        "inputs": ["object", "source"],
        "completion_predicates": ["object_grasped", "object_removed_from_source"],
        "failure_modes": ["missed_grasp", "object_slipped", "wrong_object_grasped"],
        "recovery_actions": ["retry_grasp", "reposition_gripper", "rewind_to_last_stable_object_state"],
    },
    "place_object_in_target": {
        "label": "Place object in target",
        "description": "Put a grasped object inside a receptacle or fixture.",
        "typical_triggers": ["place in", "put in", "insert", "inside"],
        "inputs": ["object", "target_receptacle"],
        "completion_predicates": ["object_inside_target", "object_supported_by_target"],
        "failure_modes": ["missed_receptacle", "object_outside_target", "object_bounced_out"],
        "recovery_actions": ["retry_place", "rewind_to_pre_place_state", "use_target-centered_prompt"],
    },
    "place_object_on_target": {
        "label": "Place object on target",
        "description": "Put a grasped object on a surface, rack, shelf, plate, or counter.",
        "typical_triggers": ["place on", "put on", "set on", "on the"],
        "inputs": ["object", "target_surface"],
        "completion_predicates": ["object_on_target", "object_touching_target"],
        "failure_modes": ["object_not_on_surface", "object_fell", "wrong_surface"],
        "recovery_actions": ["retry_place", "rewind_to_pre_place_state", "move_to_target_surface"],
    },
    "press_button": {
        "label": "Press button",
        "description": "Press a button or switch on an appliance.",
        "typical_triggers": ["press", "push", "button", "start", "stop"],
        "inputs": ["fixture", "button"],
        "completion_predicates": ["button_activated", "appliance_state_changed"],
        "failure_modes": ["button_missed", "wrong_button", "state_unchanged"],
        "recovery_actions": ["retry_press", "relocalize_button", "rewind_to_pre_press_state"],
    },
    "turn_knob": {
        "label": "Turn knob",
        "description": "Rotate a knob or dial to change appliance state.",
        "typical_triggers": ["turn", "twist", "adjust", "preheat", "temperature", "burner", "knob"],
        "inputs": ["fixture", "knob", "target_setting"],
        "completion_predicates": ["knob_at_target", "appliance_setting_reached"],
        "failure_modes": ["wrong_knob", "insufficient_rotation", "overshoot"],
        "recovery_actions": ["retry_turn", "correct_direction", "rewind_to_pre_turn_state"],
    },
    "turn_lever": {
        "label": "Turn lever",
        "description": "Move a lever or faucet/spout control.",
        "typical_triggers": ["lever", "faucet", "spout", "water"],
        "inputs": ["fixture", "lever", "target_state"],
        "completion_predicates": ["lever_at_target", "flow_or_orientation_reached"],
        "failure_modes": ["lever_missed", "wrong_direction", "state_unchanged"],
        "recovery_actions": ["retry_turn", "relocalize_lever", "rewind_to_pre_turn_state"],
    },
    "slide_rack": {
        "label": "Slide rack",
        "description": "Slide a rack or tray in or out of a fixture.",
        "typical_triggers": ["slide", "pull out", "push in", "rack", "tray"],
        "inputs": ["fixture", "rack"],
        "completion_predicates": ["rack_at_target_extension"],
        "failure_modes": ["rack_not_moved", "rack_partially_moved", "wrong_rack"],
        "recovery_actions": ["retry_slide", "regrasp_rack", "rewind_to_pre_slide_state"],
    },
    "release_and_retreat": {
        "label": "Release and retreat",
        "description": "Release the manipulated object and move the gripper away.",
        "typical_triggers": ["gripper far", "release", "done"],
        "inputs": ["object"],
        "completion_predicates": ["gripper_far_from_object", "object_stable"],
        "failure_modes": ["object_still_grasped", "object_disturbed_after_release"],
        "recovery_actions": ["open_gripper", "retreat", "rewind_to_post_place_state"],
    },
}


def get_atomic_skill_library() -> dict:
    """Return a copy of the atomic skill library."""
    return deepcopy(ATOMIC_SKILL_LIBRARY)

