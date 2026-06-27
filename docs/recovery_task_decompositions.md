# Recovery Task Decompositions

This document records the decomposition hierarchy used by recovery evaluation and failure-dataset creation.

- **HL task**: the RoboCasa environment task being attempted.
- **HL instruction**: the language instruction used to describe the full environment task.
- **Atomic task**: a dataset-level atomic action task. For atomic environments, this is the same as the HL task. For composite environments, it is either an exact registered RoboCasa atomic task or a derived task name following the same naming style.
- **Subtask**: a semantic progress step under an atomic task. One subtask may correspond to multiple runtime predicates, and the same predicate may appear in multiple semantic subtasks when the monitor does not expose finer-grained state.
- **Predicate**: the runtime boolean monitor condition backing a subtask.

Sources:

- `robocasa/recovery/eval_composite_predicates.py`: runtime predicates, HL instructions, semantic subtask grouping, and composite-to-atomic-task decompositions.
- `robocasa/recovery/create_recovery_failure_dataset.py`: serialization of atomic-task, subtask, and predicate progress into failure dataset samples.

Coverage in this checkout: 18 atomic tasks and 32 composite tasks.

## Examples

### PickPlaceDrawerToCounter

- HL task / atomic task: `PickPlaceDrawerToCounter`
- HL instruction: Pick the object from the drawer and place it on the counter.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the target object from the drawer. | `object_grasped` |
| 2 | Move the target object from the drawer to the counter. | `object_on_counter` |
| 3 | Release the target object at a valid counter location. | `final_placement_valid`, `gripper_released` |

### OpenDrawer

- HL task / atomic task: `OpenDrawer`
- HL instruction: Open the drawer.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open drawer. | `drawer_open` |

### MakeIceLemonade

- HL task: `MakeIceLemonade`
- HL instruction: Grab a lemon wedge from the fridge and ice cubes from the ice bowl, and put them in the glass of lemonade.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenFridge` | registered | Open the fridge. | `fridge_open` |
| 2 | `PickPlaceFridgeToGlass` | derived | Pick the lemon wedge from the fridge and place it in the glass of lemonade. | `lemon_grasped`, `lemon_in_glass` |
| 3 | `PickPlaceCounterToGlass` | derived | Pick the first ice cube from the ice bowl and place it in the glass of lemonade. | `ice_cube1_grasped`, `ice_in_glass` |
| 4 | `PickPlaceCounterToGlass` | derived | Pick the second ice cube from the ice bowl and place it in the glass of lemonade. | `ice_cube2_grasped`, `ice_in_glass`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open fridge. | `fridge_open` |
| 2 | Pick the lemon wedge from the fridge. | `lemon_grasped` |
| 3 | Move the lemon wedge to the glass. | `lemon_in_glass` |
| 4 | Pick the first ice cube from the ice bowl. | `ice_cube1_grasped` |
| 5 | Move the first ice cube to the glass. | `ice_in_glass` |
| 6 | Pick the second ice cube from the ice bowl. | `ice_cube2_grasped` |
| 7 | Move the second ice cube to the glass and release the ingredients. | `ice_in_glass`, `gripper_released` |

### SteamInMicrowave

- HL task: `SteamInMicrowave`
- HL instruction: Place the vegetable in the bowl, put the bowl in the microwave, close it, and start it.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceSinkToBowl` | derived | Pick the vegetable from the sink and place it in the bowl. | `vegetable_in_bowl` |
| 2 | `PickPlaceCounterToMicrowave` | registered | Pick the bowl and place it inside the microwave. | `bowl_in_microwave` |
| 3 | `CloseMicrowave` | registered | Close the microwave. | `microwave_closed` |
| 4 | `TurnOnMicrowave` | registered | Start the microwave. | `microwave_started` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the vegetable from the sink. | `vegetable_in_bowl` |
| 2 | Place the vegetable in the bowl. | `vegetable_in_bowl` |
| 3 | Release the vegetable in the bowl. | `vegetable_in_bowl` |
| 4 | Pick the bowl. | `bowl_in_microwave` |
| 5 | Place the bowl inside the microwave. | `bowl_in_microwave` |
| 6 | Release the bowl inside the microwave. | `bowl_in_microwave` |
| 7 | Close microwave. | `microwave_closed` |
| 8 | Start microwave. | `microwave_started` |

### StoreLeftoversInBowl

- HL task: `StoreLeftoversInBowl`
- HL instruction: Pick the chicken drumstick and vegetable from their plates and place them in the bowl. Then put the bowl in the fridge.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToBowl` | derived | Pick the chicken drumstick and place it in the bowl. | `chicken_in_bowl` |
| 2 | `PickPlaceCounterToBowl` | derived | Pick the vegetable and place it in the bowl. | `vegetable_in_bowl` |
| 3 | `PickPlaceBowlToFridge` | derived | Pick the bowl containing the leftovers and place it in the already-open fridge. | `bowl_in_fridge`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the chicken drumstick. | `chicken_in_bowl` |
| 2 | Place the chicken drumstick in the bowl. | `chicken_in_bowl` |
| 3 | Release the chicken drumstick in the bowl. | `chicken_in_bowl` |
| 4 | Pick the vegetable. | `vegetable_in_bowl` |
| 5 | Place the vegetable in the bowl. | `vegetable_in_bowl` |
| 6 | Release the vegetable in the bowl. | `vegetable_in_bowl` |
| 7 | Pick the bowl containing the leftovers. | `bowl_in_fridge` |
| 8 | Place the bowl containing the leftovers in the already-open fridge. | `bowl_in_fridge` |
| 9 | Release the bowl containing the leftovers in the fridge. | `bowl_in_fridge`, `gripper_released` |

### PackIdenticalLunches

- HL task: `PackIdenticalLunches`
- HL instruction: Pack matching vegetable and meat portions into two tupperware containers.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenFridge` | registered | Open the fridge. | `fridge_open` |
| 2 | `PickPlaceFridgeToTupperware` | derived | Pick the first vegetable from the fridge and place it in the first tupperware. | `tupperware0_has_one_vegetable` |
| 3 | `PickPlaceFridgeToTupperware` | derived | Pick the first meat item from the fridge and place it in the first tupperware. | `tupperware0_has_one_meat` |
| 4 | `PickPlaceFridgeToTupperware` | derived | Pick the second vegetable from the fridge and place it in the second tupperware. | `tupperware1_has_one_vegetable` |
| 5 | `PickPlaceFridgeToTupperware` | derived | Pick the second meat item from the fridge and place it in the second tupperware. | `tupperware1_has_one_meat`, `objects_not_duplicated`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the fridge. | `fridge_open` |
| 2 | Pick the first vegetable from the fridge. | `tupperware0_has_one_vegetable` |
| 3 | Place the first vegetable in the first tupperware. | `tupperware0_has_one_vegetable` |
| 4 | Pick the first meat item from the fridge. | `tupperware0_has_one_meat` |
| 5 | Place the first meat item in the first tupperware. | `tupperware0_has_one_meat` |
| 6 | Pick the second vegetable from the fridge. | `tupperware1_has_one_vegetable` |
| 7 | Place the second vegetable in the second tupperware. | `tupperware1_has_one_vegetable` |
| 8 | Pick the second meat item from the fridge. | `tupperware1_has_one_meat` |
| 9 | Place the second meat item in the second tupperware. | `tupperware1_has_one_meat` |
| 10 | Release the food after both tupperwares contain one vegetable and one meat item. | `tupperware0_has_one_vegetable`, `tupperware0_has_one_meat`, `tupperware1_has_one_vegetable`, `tupperware1_has_one_meat`, `objects_not_duplicated`, `gripper_released` |

### StirVegetables

- HL task: `StirVegetables`
- HL instruction: Place the vegetables in the pot, pick the spatula, and stir the vegetables.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToStove` | registered | Pick the first vegetable from the counter and place it in the pot. | `vegetable1_in_pot` |
| 2 | `PickPlaceCounterToStove` | registered | Pick the second vegetable from the counter and place it in the pot. | `vegetable2_in_pot` |
| 3 | `PickPlaceCounterToStove` | registered | Pick the spatula from the counter and place it in the pot. | `spatula_grasped` |
| 4 | `StirVegetables` | derived | Stir the vegetables in the pot with the spatula. | `vegetables_stirred`, `spatula_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the first vegetable from the counter. | `vegetable1_in_pot` |
| 2 | Place the first vegetable in the pot. | `vegetable1_in_pot` |
| 3 | Release the first vegetable in the pot. | `vegetable1_in_pot` |
| 4 | Pick the second vegetable from the counter. | `vegetable2_in_pot` |
| 5 | Place the second vegetable in the pot. | `vegetable2_in_pot` |
| 6 | Release the second vegetable in the pot. | `vegetable2_in_pot` |
| 7 | Pick the spatula. | `spatula_grasped` |
| 8 | Stir the vegetables in the pot with the spatula. | `vegetables_stirred`, `spatula_released` |

### GatherTableware

- HL task: `GatherTableware`
- HL instruction: Arrange the glasses together and separate the bowl from the glasses in the cabinet.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenCabinet` | registered | Open the cabinets. | `cabinets_open` |
| 2 | `PickPlaceCabinetToCabinet` | derived | Pick the third glass and place it in the cabinet with the other glasses. | `glasses_clustered` |
| 3 | `PickPlaceCabinetToCabinet` | derived | Pick the bowl and move it away from the glasses in the cabinet. | `bowl_separated_from_glasses`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the cabinets. | `cabinets_open` |
| 2 | Pick the third glass. | `glasses_clustered` |
| 3 | Place the third glass in the cabinet with the other glasses. | `glasses_clustered` |
| 4 | Release the third glass in the cabinet with the other glasses. | `glasses_clustered` |
| 5 | Pick the bowl. | `bowl_separated_from_glasses` |
| 6 | Place the bowl away from the glasses in the cabinet. | `bowl_separated_from_glasses` |
| 7 | Release the bowl away from the glasses in the cabinet. | `bowl_separated_from_glasses`, `gripper_released` |

### PanTransfer

- HL task: `PanTransfer`
- HL instruction: Transfer the vegetables from the pan to the plate, then return the pan to the stove.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceStoveToCounter` | registered | Pick the pan from the stove, keep holding it, and tilt it to dump the vegetable onto the plate. | `pan_grasped`, `vegetable_on_plate`, `robot_did_not_touch_food` |
| 2 | `PickPlaceCounterToStove` | registered | Return the pan to the stove and release it. | `pan_on_stove`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the pan from the stove. | `pan_grasped` |
| 2 | Keep holding the pan and tilt it to dump the vegetable onto the plate without touching the food. | `vegetable_on_plate`, `robot_did_not_touch_food` |
| 3 | Place the pan back on the stove and release it. | `pan_on_stove`, `gripper_released` |

### ArrangeBreadBasket

- HL task: `ArrangeBreadBasket`
- HL instruction: Open the cabinet, pick up the bread from the cabinet and place it in the basket. Then move the basket to the dining counter.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenCabinet` | registered | Open the cabinet. | `cabinet_open` |
| 2 | `PickPlaceCabinetToCounter` | registered | Pick the bread from the cabinet and place it in the basket. | `bread_in_basket` |
| 3 | `PickPlaceCounterToCounter` | derived | Pick the basket and place it on the dining counter. | `basket_on_dining_counter`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the cabinet. | `cabinet_open` |
| 2 | Pick the bread from the cabinet. | `bread_in_basket` |
| 3 | Place it in the basket. | `bread_in_basket` |
| 4 | Release the object at the target location. | `bread_in_basket` |
| 5 | Pick the basket. | `basket_on_dining_counter` |
| 6 | Place it on the dining counter. | `basket_on_dining_counter` |
| 7 | Release the object at the target location. | `basket_on_dining_counter`, `gripper_released` |

### PortionHotDogs

- HL task: `PortionHotDogs`
- HL instruction: Prepare two plates, each with one bun and one sausage.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceBowlToPlate` | derived | Pick one bun from the bowl and place it on the first plate. | `plate1_has_one_bun` |
| 2 | `PickPlaceBowlToPlate` | derived | Pick one sausage from the bowl and place it on the first plate. | `plate1_has_one_sausage` |
| 3 | `PickPlaceBowlToPlate` | derived | Pick the remaining bun from the bowl and place it on the second plate. | `plate2_has_one_bun` |
| 4 | `PickPlaceBowlToPlate` | derived | Pick the remaining sausage from the bowl and place it on the second plate. | `plate2_has_one_sausage`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick one bun from the bowl. | `plate1_has_one_bun` |
| 2 | Place the bun on the first plate. | `plate1_has_one_bun` |
| 3 | Pick one sausage from the bowl. | `plate1_has_one_sausage` |
| 4 | Place the sausage on the first plate. | `plate1_has_one_sausage` |
| 5 | Pick the remaining bun from the bowl. | `plate2_has_one_bun` |
| 6 | Place the bun on the second plate. | `plate2_has_one_bun` |
| 7 | Pick the remaining sausage from the bowl. | `plate2_has_one_sausage` |
| 8 | Place the sausage on the second plate. | `plate2_has_one_sausage` |
| 9 | Release the food after each plate has one bun and one sausage. | `plate1_has_one_bun`, `plate1_has_one_sausage`, `plate2_has_one_bun`, `plate2_has_one_sausage`, `gripper_released` |

### RecycleBottlesByType

- HL task: `RecycleBottlesByType`
- HL instruction: Group plastic bottles with plastic bottles and glass bottles with glass bottles.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToCounter` | derived | Pick the middle plastic bottle and place it with the plastic bottle group. | `plastic_bottles_clustered` |
| 2 | `PickPlaceCounterToCounter` | derived | Pick the middle glass bottle and place it with the glass bottle group. | `glass_bottles_clustered` |
| 3 | `PickPlaceCounterToCounter` | derived | Pick the mystery bottle and place it with the matching bottle group. | `plastic_bottles_clustered`, `glass_bottles_clustered`, `bottles_on_table`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the middle plastic bottle. | `plastic_bottles_clustered` |
| 2 | Place the middle plastic bottle with the plastic bottle group. | `plastic_bottles_clustered` |
| 3 | Release the middle plastic bottle with the plastic bottle group. | `plastic_bottles_clustered` |
| 4 | Pick the middle glass bottle. | `glass_bottles_clustered` |
| 5 | Place the middle glass bottle with the glass bottle group. | `glass_bottles_clustered` |
| 6 | Release the middle glass bottle with the glass bottle group. | `glass_bottles_clustered` |
| 7 | Identify the mystery bottle and place it with the matching bottle group. | `plastic_bottles_clustered`, `glass_bottles_clustered` |
| 8 | Release the bottles on the table. | `bottles_on_table`, `gripper_released` |

### SeparateFreezerRack

- HL task: `SeparateFreezerRack`
- HL instruction: Separate meat and vegetables into freezer containers and place them on the correct freezer racks.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenFreezer` | derived | Open the freezer. | `freezer_open` |
| 2 | `VerifyContainerContents` | derived | Verify the meat is in the meat container and both vegetables are in the vegetable container. | `meat_in_tupperware`, `vegetables_in_tupperware` |
| 3 | `PickPlaceCounterToFreezer` | derived | Pick the meat container and place it on the second freezer rack. | `meat_container_on_second_rack` |
| 4 | `PickPlaceCounterToFreezer` | derived | Pick the vegetable container and place it on the top freezer rack. | `vegetable_container_on_top_rack`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the freezer. | `freezer_open` |
| 2 | Verify the meat is in the meat container. | `meat_in_tupperware` |
| 3 | Verify both vegetables are in the vegetable container. | `vegetables_in_tupperware` |
| 4 | Pick the meat container. | `meat_container_on_second_rack` |
| 5 | Place the meat container on the second freezer rack. | `meat_container_on_second_rack` |
| 6 | Pick the vegetable container. | `vegetable_container_on_top_rack` |
| 7 | Place the vegetable container on the top freezer rack. | `vegetable_container_on_top_rack` |
| 8 | Release the freezer containers on their target racks. | `meat_container_on_second_rack`, `vegetable_container_on_top_rack`, `gripper_released` |

### LoadDishwasher

- HL task: `LoadDishwasher`
- HL instruction: Pull out the dishwasher rack, place the dishes on it, and close the dishwasher.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `SlideDishwasherRack` | registered | Pull out the dishwasher rack. | `dishwasher_rack_accessible` |
| 2 | `PickPlaceCounterToDishwasherRack` | derived | Pick the dishes from the counter and place them on the dishwasher rack. | `dishes_grasped`, `dishes_on_rack` |
| 3 | `CloseDishwasher` | derived | Close the dishwasher. | `dishwasher_closed` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pull out the dishwasher rack. | `dishwasher_rack_accessible` |
| 2 | Pick the cup and bowl from the counter. | `dishes_grasped` |
| 3 | Place the cup and bowl on the dishwasher rack. | `dishes_on_rack` |
| 4 | Release the cup and bowl on the dishwasher rack. | `dishes_on_rack` |
| 5 | Close the dishwasher. | `dishwasher_closed` |

### PrepareCoffee

- HL task: `PrepareCoffee`
- HL instruction: Place the mug under the coffee dispenser and start the coffee machine.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `CoffeeSetupMug` | registered | Pick the mug and place it under the coffee machine dispenser. | `mug_grasped`, `mug_under_dispenser`, `gripper_released` |
| 2 | `StartCoffeeMachine` | registered | Start the coffee machine. | `coffee_started` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the mug. | `mug_grasped` |
| 2 | Place the mug under the coffee machine dispenser. | `mug_under_dispenser` |
| 3 | Release the mug under the coffee machine dispenser. | `mug_under_dispenser`, `gripper_released` |
| 4 | Start the coffee machine. | `coffee_started` |

### StackBowlsCabinet

- HL task: `StackBowlsCabinet`
- HL instruction: Open the cabinet and stack the bowls inside it.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenCabinet` | registered | Open the cabinet. | `cabinet_open` |
| 2 | `PickPlaceCounterToCabinet` | registered | Pick the larger bowl from the counter and place it in the cabinet. | `larger_bowl_in_cabinet` |
| 3 | `PickPlaceCounterToCabinet` | registered | Pick the smaller bowl from the counter and place it in the cabinet stacked on top of the larger bowl. | `smaller_bowl_in_cabinet`, `bowls_stacked`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the cabinet. | `cabinet_open` |
| 2 | Pick the larger bowl from the counter. | `larger_bowl_in_cabinet` |
| 3 | Place the larger bowl in the cabinet. | `larger_bowl_in_cabinet` |
| 4 | Release the larger bowl in the cabinet. | `larger_bowl_in_cabinet` |
| 5 | Pick the smaller bowl from the counter. | `smaller_bowl_in_cabinet` |
| 6 | Stack the smaller bowl on top of the larger bowl in the cabinet. | `smaller_bowl_in_cabinet`, `bowls_stacked` |
| 7 | Release the smaller bowl on top of the larger bowl. | `smaller_bowl_in_cabinet`, `bowls_stacked`, `gripper_released` |

### WashLettuce

- HL task: `WashLettuce`
- HL instruction: Turn on the sink faucet and rinse the lettuce.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `TurnOnSinkFaucet` | registered | Turn on the sink faucet. | `water_on` |
| 2 | `PickPlaceCounterToSink` | registered | Move the lettuce under the running water and rinse it. | `lettuce_under_water`, `washed_time_reached` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Turn on the sink faucet. | `water_on` |
| 2 | Move the lettuce under the running water and rinse it. | `lettuce_under_water`, `washed_time_reached` |

## Atomic Tasks

### CloseBlenderLid

- HL task / atomic task: `CloseBlenderLid`
- HL instruction: Close the blender lid.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Move blender lid onto blender. | `blender_lid_on_blender` |
| 2 | Release blender lid. | `gripper_released` |

### CloseFridge

- HL task / atomic task: `CloseFridge`
- HL instruction: Close the fridge.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Close fridge. | `fridge_closed` |

### CloseToasterOvenDoor

- HL task / atomic task: `CloseToasterOvenDoor`
- HL instruction: Close the toaster oven door.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Close toaster oven. | `toaster_oven_closed` |

### CoffeeSetupMug

- HL task / atomic task: `CoffeeSetupMug`
- HL instruction: Pick the mug and place it under the coffee machine dispenser.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick mug. | `mug_grasped` |
| 2 | Move mug under coffee dispenser. | `mug_under_dispenser` |
| 3 | Release mug. | `gripper_released` |

### NavigateKitchen

- HL task / atomic task: `NavigateKitchen`
- HL instruction: Navigate to the target location in the kitchen.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Move robot to target. | `at_target_position` |
| 2 | Face target. | `facing_target` |

### OpenCabinet

- HL task / atomic task: `OpenCabinet`
- HL instruction: Open the cabinet.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open cabinet. | `cabinet_open` |

### OpenDrawer

- HL task / atomic task: `OpenDrawer`
- HL instruction: Open the drawer.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open drawer. | `drawer_open` |

### OpenStandMixerHead

- HL task / atomic task: `OpenStandMixerHead`
- HL instruction: Open the stand mixer head.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open stand mixer head. | `stand_mixer_head_open` |

### PickPlaceCounterToCabinet

- HL task / atomic task: `PickPlaceCounterToCabinet`
- HL instruction: Pick the object from the counter and place it in the cabinet.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick object from counter. | `object_grasped` |
| 2 | Move object to cabinet. | `object_in_cabinet` |
| 3 | Release object at a valid cabinet location. | `final_placement_valid`, `gripper_released` |

### PickPlaceCounterToStove

- HL task / atomic task: `PickPlaceCounterToStove`
- HL instruction: Pick the object from the counter and place it on the stove.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick object from counter. | `object_grasped` |
| 2 | Move object to pan. | `object_in_pan` |
| 3 | Release object at a valid pan location. | `final_placement_valid`, `gripper_released` |

### PickPlaceDrawerToCounter

- HL task / atomic task: `PickPlaceDrawerToCounter`
- HL instruction: Pick the object from the drawer and place it on the counter.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the target object from the drawer. | `object_grasped` |
| 2 | Move the target object from the drawer to the counter. | `object_on_counter` |
| 3 | Release the target object at a valid counter location. | `final_placement_valid`, `gripper_released` |

### PickPlaceSinkToCounter

- HL task / atomic task: `PickPlaceSinkToCounter`
- HL instruction: Pick the object from the sink and place it on the counter.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick object from sink. | `object_grasped` |
| 2 | Move object to container. | `object_in_container` |
| 3 | Move container to counter. | `container_on_counter` |
| 4 | Release object at a valid counter location. | `final_placement_valid`, `gripper_released` |

### PickPlaceToasterToCounter

- HL task / atomic task: `PickPlaceToasterToCounter`
- HL instruction: Pick the object from the toaster and place it on the counter.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick object from toaster. | `object_grasped` |
| 2 | Move object to plate. | `object_on_plate` |
| 3 | Release object at a valid plate location. | `final_placement_valid`, `gripper_released` |

### SlideDishwasherRack

- HL task / atomic task: `SlideDishwasherRack`
- HL instruction: Slide the dishwasher rack.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Slide dishwasher rack. | `dishwasher_rack_slid` |

### TurnOffStove

- HL task / atomic task: `TurnOffStove`
- HL instruction: Turn off the stove.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Turn off burner. | `burner_off` |

### TurnOnElectricKettle

- HL task / atomic task: `TurnOnElectricKettle`
- HL instruction: Turn on the electric kettle.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Turn on electric kettle. | `electric_kettle_on` |

### TurnOnMicrowave

- HL task / atomic task: `TurnOnMicrowave`
- HL instruction: Turn on the microwave.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Start microwave. | `microwave_started` |
| 2 | Release microwave button. | `gripper_released` |

### TurnOnSinkFaucet

- HL task / atomic task: `TurnOnSinkFaucet`
- HL instruction: Turn on the sink faucet.

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Turn on water. | `water_on` |

## Composite Tasks

### DeliverStraw

- HL task: `DeliverStraw`
- HL instruction: Take a straw from the drawer in front and place it inside the glass cup on the dining counter.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenDrawer` | registered | Open the drawer. | `drawer_open` |
| 2 | `PickPlaceDrawerToGlass` | derived | Pick the straw from the drawer and place it inside the glass. | `straw_grasped`, `straw_in_glass`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the drawer. | `drawer_open` |
| 2 | Pick the straw from the drawer. | `straw_grasped` |
| 3 | Place it inside the glass. | `straw_in_glass` |
| 4 | Release the object at the target location. | `straw_in_glass`, `gripper_released` |

### GetToastedBread

- HL task: `GetToastedBread`
- HL instruction: Start the toaster, then place the toasted bread on the plate.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `TurnOnToaster` | registered | Start the toaster. | `toaster_started` |
| 2 | `PickPlaceToasterToCounter` | registered | Pick the toasted bread from the toaster and place it on the plate. | `bread_grasped`, `bread_on_plate`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Start the toaster. | `toaster_started` |
| 2 | Pick the toasted bread from the toaster. | `bread_grasped` |
| 3 | Place it on the plate. | `bread_on_plate` |
| 4 | Release the object at the target location. | `bread_on_plate`, `gripper_released` |

### KettleBoiling

- HL task: `KettleBoiling`
- HL instruction: Place the kettle on the stove burner and turn on the burner.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToStove` | registered | Pick the kettle from the counter and place it on the stove burner. | `kettle_grasped`, `kettle_on_burner`, `gripper_released` |
| 2 | `TurnOnStove` | registered | Turn on the target stove burner. | `burner_on` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the kettle from the counter. | `kettle_grasped` |
| 2 | Place it on the stove burner. | `kettle_on_burner` |
| 3 | Release the object at the target location. | `kettle_on_burner`, `gripper_released` |
| 4 | Turn on the target stove burner. | `burner_on` |

### LoadDishwasher

- HL task: `LoadDishwasher`
- HL instruction: Pull out the dishwasher rack, place the dishes on it, and close the dishwasher.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `SlideDishwasherRack` | registered | Pull out the dishwasher rack. | `dishwasher_rack_accessible` |
| 2 | `PickPlaceCounterToDishwasherRack` | derived | Pick the dishes from the counter and place them on the dishwasher rack. | `dishes_grasped`, `dishes_on_rack` |
| 3 | `CloseDishwasher` | derived | Close the dishwasher. | `dishwasher_closed` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pull out the dishwasher rack. | `dishwasher_rack_accessible` |
| 2 | Pick the cup and bowl from the counter. | `dishes_grasped` |
| 3 | Place the cup and bowl on the dishwasher rack. | `dishes_on_rack` |
| 4 | Release the cup and bowl on the dishwasher rack. | `dishes_on_rack` |
| 5 | Close the dishwasher. | `dishwasher_closed` |

### MakeIceLemonade

- HL task: `MakeIceLemonade`
- HL instruction: Grab a lemon wedge from the fridge and ice cubes from the ice bowl, and put them in the glass of lemonade.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenFridge` | registered | Open the fridge. | `fridge_open` |
| 2 | `PickPlaceFridgeToGlass` | derived | Pick the lemon wedge from the fridge and place it in the glass of lemonade. | `lemon_grasped`, `lemon_in_glass` |
| 3 | `PickPlaceCounterToGlass` | derived | Pick the first ice cube from the ice bowl and place it in the glass of lemonade. | `ice_cube1_grasped`, `ice_in_glass` |
| 4 | `PickPlaceCounterToGlass` | derived | Pick the second ice cube from the ice bowl and place it in the glass of lemonade. | `ice_cube2_grasped`, `ice_in_glass`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open fridge. | `fridge_open` |
| 2 | Pick the lemon wedge from the fridge. | `lemon_grasped` |
| 3 | Move the lemon wedge to the glass. | `lemon_in_glass` |
| 4 | Pick the first ice cube from the ice bowl. | `ice_cube1_grasped` |
| 5 | Move the first ice cube to the glass. | `ice_in_glass` |
| 6 | Pick the second ice cube from the ice bowl. | `ice_cube2_grasped` |
| 7 | Move the second ice cube to the glass and release the ingredients. | `ice_in_glass`, `gripper_released` |

### PackIdenticalLunches

- HL task: `PackIdenticalLunches`
- HL instruction: Pack matching vegetable and meat portions into two tupperware containers.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenFridge` | registered | Open the fridge. | `fridge_open` |
| 2 | `PickPlaceFridgeToTupperware` | derived | Pick the first vegetable from the fridge and place it in the first tupperware. | `tupperware0_has_one_vegetable` |
| 3 | `PickPlaceFridgeToTupperware` | derived | Pick the first meat item from the fridge and place it in the first tupperware. | `tupperware0_has_one_meat` |
| 4 | `PickPlaceFridgeToTupperware` | derived | Pick the second vegetable from the fridge and place it in the second tupperware. | `tupperware1_has_one_vegetable` |
| 5 | `PickPlaceFridgeToTupperware` | derived | Pick the second meat item from the fridge and place it in the second tupperware. | `tupperware1_has_one_meat`, `objects_not_duplicated`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the fridge. | `fridge_open` |
| 2 | Pick the first vegetable from the fridge. | `tupperware0_has_one_vegetable` |
| 3 | Place the first vegetable in the first tupperware. | `tupperware0_has_one_vegetable` |
| 4 | Pick the first meat item from the fridge. | `tupperware0_has_one_meat` |
| 5 | Place the first meat item in the first tupperware. | `tupperware0_has_one_meat` |
| 6 | Pick the second vegetable from the fridge. | `tupperware1_has_one_vegetable` |
| 7 | Place the second vegetable in the second tupperware. | `tupperware1_has_one_vegetable` |
| 8 | Pick the second meat item from the fridge. | `tupperware1_has_one_meat` |
| 9 | Place the second meat item in the second tupperware. | `tupperware1_has_one_meat` |
| 10 | Release the food after both tupperwares contain one vegetable and one meat item. | `tupperware0_has_one_vegetable`, `tupperware0_has_one_meat`, `tupperware1_has_one_vegetable`, `tupperware1_has_one_meat`, `objects_not_duplicated`, `gripper_released` |

### PreSoakPan

- HL task: `PreSoakPan`
- HL instruction: Put the pan and sponge in the sink, then turn on the sink faucet.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToSink` | registered | Pick the pan from the counter and place it in the sink. | `pan_grasped`, `pan_in_sink` |
| 2 | `PickPlaceCounterToSink` | registered | Pick the sponge from the counter and place it in the sink. | `sponge_grasped`, `sponge_in_sink`, `gripper_released` |
| 3 | `TurnOnSinkFaucet` | registered | Turn on the sink faucet. | `water_on` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the pan from the counter. | `pan_grasped` |
| 2 | Place it in the sink. | `pan_in_sink` |
| 3 | Release the object at the target location. | `pan_in_sink` |
| 4 | Pick the sponge from the counter. | `sponge_grasped` |
| 5 | Place it in the sink. | `sponge_in_sink` |
| 6 | Release the object at the target location. | `sponge_in_sink`, `gripper_released` |
| 7 | Turn on the sink faucet. | `water_on` |

### PrepareCoffee

- HL task: `PrepareCoffee`
- HL instruction: Place the mug under the coffee dispenser and start the coffee machine.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `CoffeeSetupMug` | registered | Pick the mug and place it under the coffee machine dispenser. | `mug_grasped`, `mug_under_dispenser`, `gripper_released` |
| 2 | `StartCoffeeMachine` | registered | Start the coffee machine. | `coffee_started` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the mug. | `mug_grasped` |
| 2 | Place the mug under the coffee machine dispenser. | `mug_under_dispenser` |
| 3 | Release the mug under the coffee machine dispenser. | `mug_under_dispenser`, `gripper_released` |
| 4 | Start the coffee machine. | `coffee_started` |

### RinseSinkBasin

- HL task: `RinseSinkBasin`
- HL instruction: Turn on the sink faucet and rinse the sink basin.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `TurnOnSinkFaucet` | registered | Turn on the sink faucet. | `water_on` |
| 2 | `TurnSinkSpout` | registered | Move the sink spout to rinse the sink basin. | `left_basin_rinsed`, `center_basin_rinsed`, `right_basin_rinsed` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Turn on the sink faucet. | `water_on` |
| 2 | Move the sink spout to rinse the sink basin. | `left_basin_rinsed`, `center_basin_rinsed`, `right_basin_rinsed` |

### ScrubCuttingBoard

- HL task: `ScrubCuttingBoard`
- HL instruction: Pick the sponge and scrub the cutting board.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToCuttingBoard` | derived | Pick the sponge from the counter. | `sponge_grasped` |
| 2 | `ScrubCuttingBoard` | derived | Scrub the cutting board with the sponge. | `board_contact_count_reached`, `board_sweep_range_reached`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the sponge. | `sponge_grasped` |
| 2 | Scrub the cutting board with the sponge. | `board_contact_count_reached`, `board_sweep_range_reached`, `gripper_released` |

### SearingMeat

- HL task: `SearingMeat`
- HL instruction: Place the pan on the stove, place the meat in the pan, and turn on the burner.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenCabinet` | registered | Open the cabinet. | `cabinet_open` |
| 2 | `PickPlaceCabinetToStove` | derived | Pick the pan from the cabinet and place it on the target stove burner. | `pan_grasped`, `pan_on_target_burner` |
| 3 | `PickPlaceCounterToStove` | registered | Pick the meat from the counter and place it in the pan. | `meat_grasped`, `meat_in_pan`, `gripper_released` |
| 4 | `TurnOnStove` | registered | Turn on the target stove burner. | `burner_on` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the cabinet. | `cabinet_open` |
| 2 | Pick the pan from the cabinet. | `pan_grasped` |
| 3 | Place it on the target stove burner. | `pan_on_target_burner` |
| 4 | Release the object at the target location. | `pan_on_target_burner` |
| 5 | Pick the meat from the counter. | `meat_grasped` |
| 6 | Place it in the pan. | `meat_in_pan` |
| 7 | Release the object at the target location. | `meat_in_pan`, `gripper_released` |
| 8 | Turn on the target stove burner. | `burner_on` |

### SetUpCuttingStation

- HL task: `SetUpCuttingStation`
- HL instruction: Open the drawer, place the knife and meat on the cutting board.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenDrawer` | registered | Open the drawer. | `drawer_open` |
| 2 | `PickPlaceDrawerToCounter` | registered | Pick the knife from the drawer and place it on the cutting board. | `knife_grasped`, `knife_on_cutting_board` |
| 3 | `PickPlaceCounterToCounter` | derived | Pick the meat from the plate and place it on the cutting board. | `meat_grasped`, `meat_on_cutting_board`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the drawer. | `drawer_open` |
| 2 | Pick the knife from the drawer. | `knife_grasped` |
| 3 | Place it on the cutting board. | `knife_on_cutting_board` |
| 4 | Release the object at the target location. | `knife_on_cutting_board` |
| 5 | Pick the meat from the plate. | `meat_grasped` |
| 6 | Place it on the cutting board. | `meat_on_cutting_board` |
| 7 | Release the object at the target location. | `meat_on_cutting_board`, `gripper_released` |

### StackBowlsCabinet

- HL task: `StackBowlsCabinet`
- HL instruction: Open the cabinet and stack the bowls inside it.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenCabinet` | registered | Open the cabinet. | `cabinet_open` |
| 2 | `PickPlaceCounterToCabinet` | registered | Pick the larger bowl from the counter and place it in the cabinet. | `larger_bowl_in_cabinet` |
| 3 | `PickPlaceCounterToCabinet` | registered | Pick the smaller bowl from the counter and place it in the cabinet stacked on top of the larger bowl. | `smaller_bowl_in_cabinet`, `bowls_stacked`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the cabinet. | `cabinet_open` |
| 2 | Pick the larger bowl from the counter. | `larger_bowl_in_cabinet` |
| 3 | Place the larger bowl in the cabinet. | `larger_bowl_in_cabinet` |
| 4 | Release the larger bowl in the cabinet. | `larger_bowl_in_cabinet` |
| 5 | Pick the smaller bowl from the counter. | `smaller_bowl_in_cabinet` |
| 6 | Stack the smaller bowl on top of the larger bowl in the cabinet. | `smaller_bowl_in_cabinet`, `bowls_stacked` |
| 7 | Release the smaller bowl on top of the larger bowl. | `smaller_bowl_in_cabinet`, `bowls_stacked`, `gripper_released` |

### SteamInMicrowave

- HL task: `SteamInMicrowave`
- HL instruction: Place the vegetable in the bowl, put the bowl in the microwave, close it, and start it.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceSinkToBowl` | derived | Pick the vegetable from the sink and place it in the bowl. | `vegetable_in_bowl` |
| 2 | `PickPlaceCounterToMicrowave` | registered | Pick the bowl and place it inside the microwave. | `bowl_in_microwave` |
| 3 | `CloseMicrowave` | registered | Close the microwave. | `microwave_closed` |
| 4 | `TurnOnMicrowave` | registered | Start the microwave. | `microwave_started` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the vegetable from the sink. | `vegetable_in_bowl` |
| 2 | Place the vegetable in the bowl. | `vegetable_in_bowl` |
| 3 | Release the vegetable in the bowl. | `vegetable_in_bowl` |
| 4 | Pick the bowl. | `bowl_in_microwave` |
| 5 | Place the bowl inside the microwave. | `bowl_in_microwave` |
| 6 | Release the bowl inside the microwave. | `bowl_in_microwave` |
| 7 | Close microwave. | `microwave_closed` |
| 8 | Start microwave. | `microwave_started` |

### StirVegetables

- HL task: `StirVegetables`
- HL instruction: Place the vegetables in the pot, pick the spatula, and stir the vegetables.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToStove` | registered | Pick the first vegetable from the counter and place it in the pot. | `vegetable1_in_pot` |
| 2 | `PickPlaceCounterToStove` | registered | Pick the second vegetable from the counter and place it in the pot. | `vegetable2_in_pot` |
| 3 | `PickPlaceCounterToStove` | registered | Pick the spatula from the counter and place it in the pot. | `spatula_grasped` |
| 4 | `StirVegetables` | derived | Stir the vegetables in the pot with the spatula. | `vegetables_stirred`, `spatula_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the first vegetable from the counter. | `vegetable1_in_pot` |
| 2 | Place the first vegetable in the pot. | `vegetable1_in_pot` |
| 3 | Release the first vegetable in the pot. | `vegetable1_in_pot` |
| 4 | Pick the second vegetable from the counter. | `vegetable2_in_pot` |
| 5 | Place the second vegetable in the pot. | `vegetable2_in_pot` |
| 6 | Release the second vegetable in the pot. | `vegetable2_in_pot` |
| 7 | Pick the spatula. | `spatula_grasped` |
| 8 | Stir the vegetables in the pot with the spatula. | `vegetables_stirred`, `spatula_released` |

### StoreLeftoversInBowl

- HL task: `StoreLeftoversInBowl`
- HL instruction: Pick the chicken drumstick and vegetable from their plates and place them in the bowl. Then put the bowl in the fridge.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToBowl` | derived | Pick the chicken drumstick and place it in the bowl. | `chicken_in_bowl` |
| 2 | `PickPlaceCounterToBowl` | derived | Pick the vegetable and place it in the bowl. | `vegetable_in_bowl` |
| 3 | `PickPlaceBowlToFridge` | derived | Pick the bowl containing the leftovers and place it in the already-open fridge. | `bowl_in_fridge`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the chicken drumstick. | `chicken_in_bowl` |
| 2 | Place the chicken drumstick in the bowl. | `chicken_in_bowl` |
| 3 | Release the chicken drumstick in the bowl. | `chicken_in_bowl` |
| 4 | Pick the vegetable. | `vegetable_in_bowl` |
| 5 | Place the vegetable in the bowl. | `vegetable_in_bowl` |
| 6 | Release the vegetable in the bowl. | `vegetable_in_bowl` |
| 7 | Pick the bowl containing the leftovers. | `bowl_in_fridge` |
| 8 | Place the bowl containing the leftovers in the already-open fridge. | `bowl_in_fridge` |
| 9 | Release the bowl containing the leftovers in the fridge. | `bowl_in_fridge`, `gripper_released` |

### WashLettuce

- HL task: `WashLettuce`
- HL instruction: Turn on the sink faucet and rinse the lettuce.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `TurnOnSinkFaucet` | registered | Turn on the sink faucet. | `water_on` |
| 2 | `PickPlaceCounterToSink` | registered | Move the lettuce under the running water and rinse it. | `lettuce_under_water`, `washed_time_reached` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Turn on the sink faucet. | `water_on` |
| 2 | Move the lettuce under the running water and rinse it. | `lettuce_under_water`, `washed_time_reached` |

### ArrangeBreadBasket

- HL task: `ArrangeBreadBasket`
- HL instruction: Open the cabinet, pick up the bread from the cabinet and place it in the basket. Then move the basket to the dining counter.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenCabinet` | registered | Open the cabinet. | `cabinet_open` |
| 2 | `PickPlaceCabinetToCounter` | registered | Pick the bread from the cabinet and place it in the basket. | `bread_in_basket` |
| 3 | `PickPlaceCounterToCounter` | derived | Pick the basket and place it on the dining counter. | `basket_on_dining_counter`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the cabinet. | `cabinet_open` |
| 2 | Pick the bread from the cabinet. | `bread_in_basket` |
| 3 | Place it in the basket. | `bread_in_basket` |
| 4 | Release the object at the target location. | `bread_in_basket` |
| 5 | Pick the basket. | `basket_on_dining_counter` |
| 6 | Place it on the dining counter. | `basket_on_dining_counter` |
| 7 | Release the object at the target location. | `basket_on_dining_counter`, `gripper_released` |

### ArrangeTea

- HL task: `ArrangeTea`
- HL instruction: Place the kettle and mug on the tray, then close the cabinet.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToCounter` | derived | Pick the kettle and place it on the tray. | `kettle_on_tray` |
| 2 | `PickPlaceCabinetToCounter` | registered | Pick the mug from the cabinet and place it on the tray. | `mug_on_tray`, `gripper_released` |
| 3 | `CloseCabinet` | registered | Close the cabinet. | `cabinet_closed` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the kettle. | `kettle_on_tray` |
| 2 | Place it on the tray. | `kettle_on_tray` |
| 3 | Release the object at the target location. | `kettle_on_tray` |
| 4 | Pick the mug from the cabinet. | `mug_on_tray` |
| 5 | Place it on the tray. | `mug_on_tray` |
| 6 | Release the object at the target location. | `mug_on_tray`, `gripper_released` |
| 7 | Close the cabinet. | `cabinet_closed` |

### BreadSelection

- HL task: `BreadSelection`
- HL instruction: Place the croissant and jam on the cutting board.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToCounter` | derived | Pick the croissant and place it on the cutting board. | `croissant_on_cutting_board` |
| 2 | `PickPlaceCabinetToCounter` | registered | Pick the jam and place it on the cutting board. | `jam_on_cutting_board`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the croissant. | `croissant_on_cutting_board` |
| 2 | Place it on the cutting board. | `croissant_on_cutting_board` |
| 3 | Release the object at the target location. | `croissant_on_cutting_board` |
| 4 | Pick the jam. | `jam_on_cutting_board` |
| 5 | Place it on the cutting board. | `jam_on_cutting_board` |
| 6 | Release the object at the target location. | `jam_on_cutting_board`, `gripper_released` |

### CategorizeCondiments

- HL task: `CategorizeCondiments`
- HL instruction: Place each condiment next to its matching counterpart in the cabinet.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToCabinet` | registered | Pick the condiment bottle and place it next to the matching bottle in the cabinet. | `bottle_in_cabinet`, `bottle_next_to_counterpart` |
| 2 | `PickPlaceCounterToCabinet` | registered | Pick the shaker and place it next to the matching shaker in the cabinet. | `shaker_in_cabinet`, `shaker_next_to_counterpart`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the condiment bottle. | `bottle_in_cabinet` |
| 2 | Place it next to the matching bottle in the cabinet. | `bottle_in_cabinet`, `bottle_next_to_counterpart` |
| 3 | Release the object at the target location. | `bottle_next_to_counterpart` |
| 4 | Pick the shaker. | `shaker_in_cabinet` |
| 5 | Place it next to the matching shaker in the cabinet. | `shaker_in_cabinet`, `shaker_next_to_counterpart` |
| 6 | Release the object at the target location. | `shaker_in_cabinet`, `shaker_next_to_counterpart`, `gripper_released` |

### CuttingToolSelection

- HL task: `CuttingToolSelection`
- HL instruction: Open the drawer, select the correct cutting tool, and place it on the cutting board.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenDrawer` | registered | Open the drawer. | `drawer_open` |
| 2 | `PickPlaceDrawerToCounter` | registered | Pick the correct cutting tool from the drawer and place it on the cutting board. | `correct_tool_grasped`, `correct_tool_on_cutting_board`, `wrong_tool_not_on_cutting_board`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the drawer. | `drawer_open` |
| 2 | Pick the correct cutting tool from the drawer. | `correct_tool_grasped` |
| 3 | Place it on the cutting board. | `correct_tool_on_cutting_board`, `wrong_tool_not_on_cutting_board` |
| 4 | Release the object at the target location. | `correct_tool_on_cutting_board`, `wrong_tool_not_on_cutting_board`, `gripper_released` |

### GarnishPancake

- HL task: `GarnishPancake`
- HL instruction: Open the fridge, pick the strawberry, and place it on the pancake.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenFridge` | registered | Open the fridge. | `fridge_open` |
| 2 | `PickPlaceFridgeToCounter` | derived | Pick the strawberry from the fridge and place it on the pancake. | `strawberry_grasped`, `strawberry_on_pancake`, `gripper_released` |
| 3 | `PickPlaceCounterToCounter` | derived | Place the pancake on the plate and move the plate to the table. | `pancake_on_plate`, `plate_on_table` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the fridge. | `fridge_open` |
| 2 | Pick the strawberry from the fridge. | `strawberry_grasped` |
| 3 | Place the strawberry on the pancake. | `strawberry_on_pancake` |
| 4 | Release the strawberry on the pancake. | `strawberry_on_pancake`, `gripper_released` |
| 5 | Place the pancake on the plate. | `pancake_on_plate` |
| 6 | Move the plate with the pancake to the table. | `plate_on_table` |

### GatherTableware

- HL task: `GatherTableware`
- HL instruction: Arrange the glasses together and separate the bowl from the glasses in the cabinet.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenCabinet` | registered | Open the cabinets. | `cabinets_open` |
| 2 | `PickPlaceCabinetToCabinet` | derived | Pick the third glass and place it in the cabinet with the other glasses. | `glasses_clustered` |
| 3 | `PickPlaceCabinetToCabinet` | derived | Pick the bowl and move it away from the glasses in the cabinet. | `bowl_separated_from_glasses`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the cabinets. | `cabinets_open` |
| 2 | Pick the third glass. | `glasses_clustered` |
| 3 | Place the third glass in the cabinet with the other glasses. | `glasses_clustered` |
| 4 | Release the third glass in the cabinet with the other glasses. | `glasses_clustered` |
| 5 | Pick the bowl. | `bowl_separated_from_glasses` |
| 6 | Place the bowl away from the glasses in the cabinet. | `bowl_separated_from_glasses` |
| 7 | Release the bowl away from the glasses in the cabinet. | `bowl_separated_from_glasses`, `gripper_released` |

### HeatKebabSandwich

- HL task: `HeatKebabSandwich`
- HL instruction: Place the kebab and baguette in the toaster oven, close it, and set the timer.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToToasterOven` | registered | Pick the kebab and place it inside the toaster oven. | `kebab_in_toaster_oven` |
| 2 | `PickPlaceCounterToToasterOven` | registered | Pick the baguette and place it inside the toaster oven. | `baguette_in_toaster_oven` |
| 3 | `CloseToasterOvenDoor` | registered | Close the toaster oven door. | `toaster_oven_closed` |
| 4 | `TurnOnToasterOven` | registered | Set the toaster oven timer. | `timer_set` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the kebab. | `kebab_in_toaster_oven` |
| 2 | Place it inside the toaster oven. | `kebab_in_toaster_oven` |
| 3 | Release the object at the target location. | `kebab_in_toaster_oven` |
| 4 | Pick the baguette. | `baguette_in_toaster_oven` |
| 5 | Place it inside the toaster oven. | `baguette_in_toaster_oven` |
| 6 | Release the object at the target location. | `baguette_in_toaster_oven` |
| 7 | Close the toaster oven door. | `toaster_oven_closed` |
| 8 | Set the toaster oven timer. | `timer_set` |

### PanTransfer

- HL task: `PanTransfer`
- HL instruction: Transfer the vegetables from the pan to the plate, then return the pan to the stove.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceStoveToCounter` | registered | Pick the pan from the stove, keep holding it, and tilt it to dump the vegetable onto the plate. | `pan_grasped`, `vegetable_on_plate`, `robot_did_not_touch_food` |
| 2 | `PickPlaceCounterToStove` | registered | Return the pan to the stove and release it. | `pan_on_stove`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the pan from the stove. | `pan_grasped` |
| 2 | Keep holding the pan and tilt it to dump the vegetable onto the plate without touching the food. | `vegetable_on_plate`, `robot_did_not_touch_food` |
| 3 | Place the pan back on the stove and release it. | `pan_on_stove`, `gripper_released` |

### PortionHotDogs

- HL task: `PortionHotDogs`
- HL instruction: Prepare two plates, each with one bun and one sausage.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceBowlToPlate` | derived | Pick one bun from the bowl and place it on the first plate. | `plate1_has_one_bun` |
| 2 | `PickPlaceBowlToPlate` | derived | Pick one sausage from the bowl and place it on the first plate. | `plate1_has_one_sausage` |
| 3 | `PickPlaceBowlToPlate` | derived | Pick the remaining bun from the bowl and place it on the second plate. | `plate2_has_one_bun` |
| 4 | `PickPlaceBowlToPlate` | derived | Pick the remaining sausage from the bowl and place it on the second plate. | `plate2_has_one_sausage`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick one bun from the bowl. | `plate1_has_one_bun` |
| 2 | Place the bun on the first plate. | `plate1_has_one_bun` |
| 3 | Pick one sausage from the bowl. | `plate1_has_one_sausage` |
| 4 | Place the sausage on the first plate. | `plate1_has_one_sausage` |
| 5 | Pick the remaining bun from the bowl. | `plate2_has_one_bun` |
| 6 | Place the bun on the second plate. | `plate2_has_one_bun` |
| 7 | Pick the remaining sausage from the bowl. | `plate2_has_one_sausage` |
| 8 | Place the sausage on the second plate. | `plate2_has_one_sausage` |
| 9 | Release the food after each plate has one bun and one sausage. | `plate1_has_one_bun`, `plate1_has_one_sausage`, `plate2_has_one_bun`, `plate2_has_one_sausage`, `gripper_released` |

### RecycleBottlesByType

- HL task: `RecycleBottlesByType`
- HL instruction: Group plastic bottles with plastic bottles and glass bottles with glass bottles.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToCounter` | derived | Pick the middle plastic bottle and place it with the plastic bottle group. | `plastic_bottles_clustered` |
| 2 | `PickPlaceCounterToCounter` | derived | Pick the middle glass bottle and place it with the glass bottle group. | `glass_bottles_clustered` |
| 3 | `PickPlaceCounterToCounter` | derived | Pick the mystery bottle and place it with the matching bottle group. | `plastic_bottles_clustered`, `glass_bottles_clustered`, `bottles_on_table`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the middle plastic bottle. | `plastic_bottles_clustered` |
| 2 | Place the middle plastic bottle with the plastic bottle group. | `plastic_bottles_clustered` |
| 3 | Release the middle plastic bottle with the plastic bottle group. | `plastic_bottles_clustered` |
| 4 | Pick the middle glass bottle. | `glass_bottles_clustered` |
| 5 | Place the middle glass bottle with the glass bottle group. | `glass_bottles_clustered` |
| 6 | Release the middle glass bottle with the glass bottle group. | `glass_bottles_clustered` |
| 7 | Identify the mystery bottle and place it with the matching bottle group. | `plastic_bottles_clustered`, `glass_bottles_clustered` |
| 8 | Release the bottles on the table. | `bottles_on_table`, `gripper_released` |

### SeparateFreezerRack

- HL task: `SeparateFreezerRack`
- HL instruction: Separate meat and vegetables into freezer containers and place them on the correct freezer racks.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenFreezer` | derived | Open the freezer. | `freezer_open` |
| 2 | `VerifyContainerContents` | derived | Verify the meat is in the meat container and both vegetables are in the vegetable container. | `meat_in_tupperware`, `vegetables_in_tupperware` |
| 3 | `PickPlaceCounterToFreezer` | derived | Pick the meat container and place it on the second freezer rack. | `meat_container_on_second_rack` |
| 4 | `PickPlaceCounterToFreezer` | derived | Pick the vegetable container and place it on the top freezer rack. | `vegetable_container_on_top_rack`, `gripper_released` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the freezer. | `freezer_open` |
| 2 | Verify the meat is in the meat container. | `meat_in_tupperware` |
| 3 | Verify both vegetables are in the vegetable container. | `vegetables_in_tupperware` |
| 4 | Pick the meat container. | `meat_container_on_second_rack` |
| 5 | Place the meat container on the second freezer rack. | `meat_container_on_second_rack` |
| 6 | Pick the vegetable container. | `vegetable_container_on_top_rack` |
| 7 | Place the vegetable container on the top freezer rack. | `vegetable_container_on_top_rack` |
| 8 | Release the freezer containers on their target racks. | `meat_container_on_second_rack`, `vegetable_container_on_top_rack`, `gripper_released` |

### WaffleReheat

- HL task: `WaffleReheat`
- HL instruction: Put the waffle in the bowl, place the bowl in the microwave, close it, and start it.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `OpenMicrowave` | registered | Open the microwave. | `microwave_open` |
| 2 | `PickPlaceCounterToBowl` | derived | Pick the waffle and place it in the bowl. | `waffle_in_bowl` |
| 3 | `PickPlaceCounterToMicrowave` | registered | Pick the bowl and place it inside the microwave. | `bowl_in_microwave` |
| 4 | `CloseMicrowave` | registered | Close the microwave. | `microwave_closed` |
| 5 | `TurnOnMicrowave` | registered | Start the microwave. | `microwave_started` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Open the microwave. | `microwave_open` |
| 2 | Pick the waffle. | `waffle_in_bowl` |
| 3 | Place it in the bowl. | `waffle_in_bowl` |
| 4 | Release the object at the target location. | `waffle_in_bowl` |
| 5 | Pick the bowl. | `bowl_in_microwave` |
| 6 | Place it inside the microwave. | `bowl_in_microwave` |
| 7 | Release the object at the target location. | `bowl_in_microwave` |
| 8 | Close the microwave. | `microwave_closed` |
| 9 | Start the microwave. | `microwave_started` |

### WashFruitColander

- HL task: `WashFruitColander`
- HL instruction: Place the colander in the sink, put the fruit in the colander, and rinse it.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCounterToSink` | registered | Pick the colander from the counter and place it in the sink. | `colander_in_sink` |
| 2 | `PickPlaceCounterToSink` | registered | Pick the fruit from the counter and place it in the colander. | `fruit_in_colander` |
| 3 | `TurnOnSinkFaucet` | registered | Move the colander under the sink water. | `colander_under_water` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the colander from the counter. | `colander_in_sink` |
| 2 | Place it in the sink. | `colander_in_sink` |
| 3 | Release the object at the target location. | `colander_in_sink` |
| 4 | Pick the fruit from the counter. | `fruit_in_colander` |
| 5 | Place it in the colander. | `fruit_in_colander` |
| 6 | Release the object at the target location. | `fruit_in_colander` |
| 7 | Move the colander under the sink water. | `colander_under_water` |

### WeighIngredients

- HL task: `WeighIngredients`
- HL instruction: Place the packaged food on the scale and close the cabinet.

Atomic-task decomposition:

| # | Atomic task | Source | Language instruction | Predicate(s) |
|---:|---|---|---|---|
| 1 | `PickPlaceCabinetToCounter` | registered | Pick the packaged food from the cabinet and place it on the scale. | `packaged_food_grasped`, `packaged_food_on_scale`, `packaged_food_upright`, `gripper_released` |
| 2 | `CloseCabinet` | registered | Close the cabinet. | `cabinet_closed` |

Subtask decomposition:

| # | Subtask | Predicate(s) |
|---:|---|---|
| 1 | Pick the packaged food from the cabinet. | `packaged_food_grasped` |
| 2 | Place it on the scale. | `packaged_food_on_scale`, `packaged_food_upright` |
| 3 | Release the object at the target location. | `packaged_food_on_scale`, `packaged_food_upright`, `gripper_released` |
| 4 | Close the cabinet. | `cabinet_closed` |
