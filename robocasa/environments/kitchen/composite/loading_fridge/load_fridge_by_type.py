from robocasa.environments.kitchen.kitchen import *


class LoadFridgeByType(Kitchen):
    """
    Load Fridge By Type: composite task for Loading Fridge.

    Simulates the task of placing vegetables and meat items from the counter into
    specific shelves of the fridge based on food type, organized in bowls.

    Steps:
        Place a bowl of vegetables and a bowl of meat on different shelves of the fridge.
    """

    # certain fridge-side-by-side fridges only have 1 shelf
    EXCLUDE_STYLES = [11, 15, 18, 22, 34, 45, 49, 52, 53, 54]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.fridge, full_depth_region=True),
        )

        if "refs" in self._ep_meta:
            self.meat_rack_index = self._ep_meta["refs"]["meat_rack_index"]
            self.veg_rack_index = self._ep_meta["refs"]["veg_rack_index"]
        else:
            self.meat_rack_index = -1 if self.rng.random() < 0.5 else -2
            self.veg_rack_index = -2 if self.meat_rack_index == -1 else -1

        self.init_robot_base_ref = self.counter

    def _setup_scene(self):
        self.fridge.open_door(env=self)
        super()._setup_scene()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veg1 = self.get_obj_lang("vegetable1")
        veg2 = self.get_obj_lang("vegetable2")
        meat1 = self.get_obj_lang("meat1")
        meat2 = self.get_obj_lang("meat2")

        if veg1 == veg2:
            veg_text = f"{veg1}s"
        else:
            veg_text = f"{veg1} and {veg2}"

        if meat1 == meat2:
            meat_text = f"{meat1}s"
        else:
            meat_text = f"{meat1} and {meat2}"

        if self.meat_rack_index == -1:
            meat_shelf = "top shelf"
            veg_shelf = "second highest shelf"
        else:
            meat_shelf = "second highest shelf"
            veg_shelf = "top shelf"

        ep_meta["lang"] = (
            f"Place the bowl with the {veg_text} on the {veg_shelf} of the fridge. "
            f"Place the bowl with the {meat_text} on the {meat_shelf} of the fridge."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["meat_rack_index"] = self.meat_rack_index
        ep_meta["refs"]["veg_rack_index"] = self.veg_rack_index
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="veg_bowl",
                obj_groups="bowl",
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        full_depth_region=True,
                    ),
                    size=(1.0, 0.40),
                    pos=(-0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat_bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="veg_bowl",
                    size=(1.0, 0.40),
                    pos=(0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                graspable=True,
                fridgable=True,
                placement=dict(
                    object="veg_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups="vegetable",
                graspable=True,
                fridgable=True,
                placement=dict(
                    object="veg_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat1",
                obj_groups="meat",
                exclude_obj_groups=("shrimp"),
                graspable=True,
                fridgable=True,
                placement=dict(
                    object="meat_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat2",
                obj_groups="meat",
                exclude_obj_groups=("shrimp"),
                graspable=True,
                fridgable=True,
                placement=dict(
                    object="meat_bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_task_predicates(self):
        veg1_in_bowl = OU.check_obj_in_receptacle(self, "vegetable1", "veg_bowl")
        veg2_in_bowl = OU.check_obj_in_receptacle(self, "vegetable2", "veg_bowl")
        meat1_in_bowl = OU.check_obj_in_receptacle(self, "meat1", "meat_bowl")
        meat2_in_bowl = OU.check_obj_in_receptacle(self, "meat2", "meat_bowl")

        vegetable_bowl_on_target_shelf = self.fridge.check_rack_contact(
            self, "veg_bowl", rack_index=self.veg_rack_index, compartment="fridge"
        )
        meat_bowl_on_target_shelf = self.fridge.check_rack_contact(
            self, "meat_bowl", rack_index=self.meat_rack_index, compartment="fridge"
        )
        gripper_released = all(
            OU.gripper_obj_far(self, obj) for obj in ["veg_bowl", "meat_bowl"]
        )

        return {
            "vegetables_in_vegetable_bowl": veg1_in_bowl and veg2_in_bowl,
            "meats_in_meat_bowl": meat1_in_bowl and meat2_in_bowl,
            "vegetable_bowl_on_target_shelf": vegetable_bowl_on_target_shelf,
            "meat_bowl_on_target_shelf": meat_bowl_on_target_shelf,
            "gripper_released": gripper_released,
        }

    def _check_success(self):
        predicates = self._check_task_predicates()
        return (
            predicates["vegetables_in_vegetable_bowl"]
            and predicates["meats_in_meat_bowl"]
            and predicates["vegetable_bowl_on_target_shelf"]
            and predicates["meat_bowl_on_target_shelf"]
            and predicates["gripper_released"]
        )

    def _safe_check_obj_grasped(self, obj_name):
        try:
            return OU.check_obj_grasped(self, obj_name)
        except Exception:
            return False

    def _check_subtask_predicates(self):
        predicates = self._check_task_predicates()

        return {
            "fridge_open": {
                "value": self.fridge.is_open(self),
                "required": False,
                "source": "fixture_state",
                "stage": "precondition",
            },
            "vegetables_in_vegetable_bowl": {
                "value": predicates["vegetables_in_vegetable_bowl"],
                "required": True,
                "source": "_check_success",
                "stage": "object_state",
            },
            "meats_in_meat_bowl": {
                "value": predicates["meats_in_meat_bowl"],
                "required": True,
                "source": "_check_success",
                "stage": "object_state",
            },
            "vegetable_bowl_grasped": {
                "value": self._safe_check_obj_grasped("veg_bowl"),
                "required": False,
                "source": "diagnostic",
                "stage": "transient",
            },
            "meat_bowl_grasped": {
                "value": self._safe_check_obj_grasped("meat_bowl"),
                "required": False,
                "source": "diagnostic",
                "stage": "transient",
            },
            "vegetable_bowl_on_target_shelf": {
                "value": predicates["vegetable_bowl_on_target_shelf"],
                "required": True,
                "source": "_check_success",
                "stage": "placement",
            },
            "meat_bowl_on_target_shelf": {
                "value": predicates["meat_bowl_on_target_shelf"],
                "required": True,
                "source": "_check_success",
                "stage": "placement",
            },
            "gripper_released": {
                "value": predicates["gripper_released"],
                "required": True,
                "source": "_check_success",
                "stage": "release",
            },
        }
