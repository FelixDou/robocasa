# Recovery failure dataset analysis

- Dataset type: `recovery_failure_dataset`
- Schema version: `1`
- Samples analyzed: **1,500**
- Tasks covered: **50**
- Atomic samples: **540**
- Composite samples: **960**

## Most frequent failed subtasks

| task_name            | subtask_instruction    | failures | opportunities | failure_rate |
| -------------------- | ---------------------- | -------- | ------------- | ------------ |
| ArrangeTea           | Pick the kettle.       | 30       | 30            | 100.0%       |
| CloseFridge          | Close fridge.          | 30       | 30            | 100.0%       |
| CloseToasterOvenDoor | Close toaster oven.    | 30       | 30            | 100.0%       |
| CuttingToolSelection | Open the drawer.       | 30       | 30            | 100.0%       |
| DeliverStraw         | Open the drawer.       | 30       | 30            | 100.0%       |
| GatherTableware      | Pick the third glass.  | 30       | 30            | 100.0%       |
| NavigateKitchen      | Move robot to target.  | 30       | 30            | 100.0%       |
| OpenCabinet          | Open cabinet.          | 30       | 30            | 100.0%       |
| OpenDrawer           | Open drawer.           | 30       | 30            | 100.0%       |
| OpenStandMixerHead   | Open stand mixer head. | 30       | 30            | 100.0%       |

## Least progress before failure

| task_name            | task_granularity | samples | mean_ordered_subtask_progress |
| -------------------- | ---------------- | ------- | ----------------------------- |
| SeparateFreezerRack  | composite        | 30      | 0.0%                          |
| ArrangeTea           | composite        | 30      | 0.0%                          |
| TurnOnSinkFaucet     | atomic           | 30      | 0.0%                          |
| TurnOnElectricKettle | atomic           | 30      | 0.0%                          |
| TurnOffStove         | atomic           | 30      | 0.0%                          |
| SlideDishwasherRack  | atomic           | 30      | 0.0%                          |
| SetUpCuttingStation  | composite        | 30      | 0.0%                          |
| DeliverStraw         | composite        | 30      | 0.0%                          |
| PanTransfer          | composite        | 30      | 0.0%                          |
| CuttingToolSelection | composite        | 30      | 0.0%                          |

## Most progress before failure

| task_name                | task_granularity | samples | mean_ordered_subtask_progress |
| ------------------------ | ---------------- | ------- | ----------------------------- |
| SteamInMicrowave         | composite        | 30      | 37.5%                         |
| PreSoakPan               | composite        | 30      | 33.3%                         |
| StackBowlsCabinet        | composite        | 30      | 28.1%                         |
| LoadDishwasher           | composite        | 30      | 27.8%                         |
| StirVegetables           | composite        | 30      | 21.4%                         |
| SearingMeat              | composite        | 30      | 18.9%                         |
| PickPlaceDrawerToCounter | atomic           | 30      | 18.3%                         |
| PackIdenticalLunches     | composite        | 30      | 16.7%                         |
| GatherTableware          | composite        | 30      | 14.3%                         |
| ArrangeBreadBasket       | composite        | 30      | 13.8%                         |

## Interpretation notes

- Failure-mode counts are multi-label and can exceed the number of samples.
- Ordered subtask progress counts only the contiguous completed prefix, preserving sequentiality.
- Failed-subtask rates use samples containing that task-subtask pair as the denominator.
- The dataset contains failed rollouts, so progress is not a task success rate.
- Failures recorded at the full rollout horizon: **100.0%**.
- Composite samples with more than one stored atomic step: **0.0%**.
