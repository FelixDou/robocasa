# RoboCasa recovery results


## Mode-level summary


| Mode | HL success | Tried-subtask acc. | Avg successful subtasks before failure | Recovery success / attempt | Recovery success / all trials |
|---|---|---|---|---|---|
| continue | 30/250 (12.0%) | 201/421 (47.7%) | 0.50 | 5/220 (2.3%) | 5/250 (2.0%) |
| eef_to_last_good | 37/250 (14.8%) | 199/410 (48.5%) | 0.51 | 12/213 (5.6%) | 12/250 (4.8%) |
| env_to_last_good | 32/250 (12.8%) | 192/410 (46.8%) | 0.47 | 40/218 (18.3%) | 40/250 (16.0%) |


## Trial outcomes


| Mode | Success no recovery | Failure + recovery success | Failure + recovery failed |
|---|---|---|---|
| continue | 30/250 (12.0%) | 5/250 (2.0%) | 215/250 (86.0%) |
| eef_to_last_good | 37/250 (14.8%) | 12/250 (4.8%) | 201/250 (80.4%) |
| env_to_last_good | 32/250 (12.8%) | 40/250 (16.0%) | 178/250 (71.2%) |


## Recovery after prior progress


| Mode | Attempts with prior progress | Success |
|---|---|---|
| continue | 76 | 4/76 (5.3%) |
| eef_to_last_good | 71 | 8/71 (11.3%) |
| env_to_last_good | 74 | 16/74 (21.6%) |


## Most successful recovery subtasks


### continue


| Subtask | Success | Attempts | Accuracy |
|---|---|---|---|
| Pick pan. | 1 | 13 | 7.7% |
| Pick mug. | 1 | 9 | 11.1% |
| Open fridge. | 1 | 8 | 12.5% |
| Pick kettle. | 1 | 5 | 20.0% |
| Move vegetable to first tupperware. | 1 | 2 | 50.0% |
| Open drawer. | 0 | 20 | 0.0% |
| Pick object. | 0 | 12 | 0.0% |
| Turn on water. | 0 | 12 | 0.0% |

### eef_to_last_good


| Subtask | Success | Attempts | Accuracy |
|---|---|---|---|
| Pick pan. | 2 | 14 | 14.3% |
| Pick kettle. | 2 | 4 | 50.0% |
| Pick mug. | 1 | 9 | 11.1% |
| Open cabinet. | 1 | 8 | 12.5% |
| Turn on water. | 1 | 8 | 12.5% |
| Move colander to sink. | 1 | 5 | 20.0% |
| Open microwave. | 1 | 5 | 20.0% |
| Pick dishes. | 1 | 5 | 20.0% |

### env_to_last_good


| Subtask | Success | Attempts | Accuracy |
|---|---|---|---|
| Turn on water. | 5 | 10 | 50.0% |
| Pick pan. | 4 | 7 | 57.1% |
| Pick sponge. | 4 | 4 | 100.0% |
| Open fridge. | 3 | 11 | 27.3% |
| Open cabinet. | 3 | 9 | 33.3% |
| Open microwave. | 3 | 4 | 75.0% |
| Release pan. | 2 | 2 | 100.0% |
| Move mug under coffee dispenser. | 1 | 10 | 10.0% |


## Most failed recovery subtasks


### continue


| Subtask | Failures | Attempts | Failure rate |
|---|---|---|---|
| Open drawer. | 20 | 20 | 100.0% |
| Pick pan. | 12 | 13 | 92.3% |
| Pick object. | 12 | 12 | 100.0% |
| Turn on water. | 12 | 12 | 100.0% |
| Open cabinet. | 9 | 9 | 100.0% |
| Pick mug. | 8 | 9 | 88.9% |
| Open fridge. | 7 | 8 | 87.5% |
| Close toaster oven. | 5 | 5 | 100.0% |

### eef_to_last_good


| Subtask | Failures | Attempts | Failure rate |
|---|---|---|---|
| Open drawer. | 17 | 17 | 100.0% |
| Pick pan. | 12 | 14 | 85.7% |
| Open fridge. | 11 | 11 | 100.0% |
| Pick object. | 11 | 11 | 100.0% |
| Pick mug. | 8 | 9 | 88.9% |
| Open cabinet. | 7 | 8 | 87.5% |
| Turn on water. | 7 | 8 | 87.5% |
| Group glasses. | 5 | 5 | 100.0% |

### env_to_last_good


| Subtask | Failures | Attempts | Failure rate |
|---|---|---|---|
| Open drawer. | 18 | 18 | 100.0% |
| Move mug under coffee dispenser. | 9 | 10 | 90.0% |
| Open fridge. | 8 | 11 | 72.7% |
| Open cabinet. | 6 | 9 | 66.7% |
| Turn on water. | 5 | 10 | 50.0% |
| Group glasses. | 5 | 5 | 100.0% |
| Group plastic bottles. | 5 | 5 | 100.0% |
| Move blender lid onto blender. | 5 | 5 | 100.0% |
