# Derived RoboCasa Composite Subtask Plans

This file summarizes heuristic recovery-oriented plans. These are not official RoboCasa subtask annotations.

- Composite tasks processed: 300
- Confidence: high=113, medium=183, low=4
- Atomic skill templates: open_fixture, close_fixture, pick_object, place_object_in_target, place_object_on_target, press_button, turn_knob, turn_lever, slide_rack, release_and_retreat

## Example Plans

### MakeIceLemonade

- Description: Grab a lemon wedge from the fridge and one ice cube from the ice bowl, and put them in the glass of lemonade.
- Declared `num_subtasks`: 5
- Docs skill tags: PickPlace, door_open, nav
- Source: `robocasa/environments/kitchen/composite/adding_ice_to_beverages/make_ice_lemonade.py`
- Confidence: high

| # | Skill | Evidence text | Source |
|---:|---|---|---|
| 1 | `open_fixture` | Open the fridge to access the lemon wedge. | curated_eval_override |
| 2 | `pick_object` | Pick the lemon wedge from the fridge. | curated_eval_override |
| 3 | `place_object_in_target` | Place the lemon wedge in the glass of lemonade. | curated_eval_override |
| 4 | `pick_object` | Pick one ice cube from the ice bowl. | curated_eval_override |
| 5 | `place_object_in_target` | Place the ice cube in the glass of lemonade. | curated_eval_override |
| 6 | `release_and_retreat` | Release the last object and move the gripper away from the glass. | curated_eval_override |

### LoadFridgeByType

- Description: Place the bowl with the [*vegetable*] on the *top/second highest* shelf of the fridge. Place the bowl with the [*meat*] on the *second highest/top* highest shelf of the fridge.
- Declared `num_subtasks`: 7
- Docs skill tags: PickPlace, nav
- Source: `robocasa/environments/kitchen/composite/loading_fridge/load_fridge_by_type.py`
- Confidence: medium

| # | Skill | Evidence text | Source |
|---:|---|---|---|
| 1 | `pick_object` | Place the bowl with the [*vegetable*] on the *top/second highest* shelf of the fridge. | task_description |
| 2 | `place_object_on_target` | Place the bowl with the [*vegetable*] on the *top/second highest* shelf of the fridge. | task_description |
| 3 | `pick_object` | Place the bowl with the [*meat*] on the *second highest/top* highest shelf of the fridge. | task_description |
| 4 | `place_object_on_target` | Place the bowl with the [*meat*] on the *second highest/top* highest shelf of the fridge. | task_description |
| 5 | `release_and_retreat` | Release manipulated objects and retreat until gripper-far predicates are satisfied. | success_predicate |

Gaps:
- No source docstring Steps: block found; plan inferred from task description only.

### RecycleBottlesByType

- Description: Move the plastic bottles in the middle to the plastics group, and the glass bottles in the middle to the glass group.
- Declared `num_subtasks`: 3
- Docs skill tags: PickPlace, nav
- Source: `robocasa/environments/kitchen/composite/organizing_recycling/recycle_bottles_by_type.py`
- Confidence: medium

| # | Skill | Evidence text | Source |
|---:|---|---|---|
| 1 | `pick_object` | Pick the plastic bottle from the middle group. | curated_eval_override |
| 2 | `place_object_on_target` | Place the plastic bottle with the plastics group. | curated_eval_override |
| 3 | `pick_object` | Pick the glass bottle from the middle group. | curated_eval_override |
| 4 | `place_object_on_target` | Place the glass bottle with the glass group. | curated_eval_override |
| 5 | `release_and_retreat` | Release the last bottle and move the gripper away from the recycling groups. | curated_eval_override |

### PrepareCoffee

- Description: Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button.
- Declared `num_subtasks`: 2
- Docs skill tags: PickPlace, button_press
- Source: `robocasa/environments/kitchen/composite/brewing/prepare_coffee.py`
- Confidence: medium

| # | Skill | Evidence text | Source |
|---:|---|---|---|
| 1 | `pick_object` | Pick the mug from the open cabinet. | curated_eval_override |
| 2 | `place_object_on_target` | Place the mug under the coffee machine dispenser. | curated_eval_override |
| 3 | `press_button` | Press the coffee machine start button. | curated_eval_override |
| 4 | `release_and_retreat` | Move the gripper away from the mug and start button. | curated_eval_override |

### StackBowlsCabinet

- Description: Pick up the bowls on the counter and stack them on top of one another in the open cabinet. Place the smaller bowl on top of the larger bowl.
- Declared `num_subtasks`: 2
- Docs skill tags: PickPlace, nav
- Source: `robocasa/environments/kitchen/composite/organizing_dishes_and_containers/stack_bowls_cabinet.py`
- Confidence: medium

| # | Skill | Evidence text | Source |
|---:|---|---|---|
| 1 | `pick_object` | Pick the larger bowl from the counter. | curated_eval_override |
| 2 | `place_object_in_target` | Place the larger bowl in the open cabinet. | curated_eval_override |
| 3 | `pick_object` | Pick the smaller bowl from the counter. | curated_eval_override |
| 4 | `place_object_on_target` | Stack the smaller bowl on top of the larger bowl inside the cabinet. | curated_eval_override |
| 5 | `release_and_retreat` | Release the bowl stack and move the gripper away from the cabinet. | curated_eval_override |
