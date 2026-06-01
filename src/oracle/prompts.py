"""Positive and negative textual descriptions for pig behaviors."""

from typing import Dict, List, Optional

class DirectLabelMapper:
    """24 个标签的直接正/负样本描述映射器"""

    def __init__(self):
        self.label_descriptions = {
            "eating": {
                "positive": [
                    "a pig eating food from a trough",
                    "a pig feeding at the feeder",
                    "a pig with snout in the trough consuming feed",
                    "a pig foraging and eating at feeding area",
                    "a pig actively consuming feed",
                    "a pig with head down eating from feeder",
                ],
                "negative": [
                    "a pig drinking water from nipple drinker",
                    "a pig standing without eating",
                    "a pig lying down resting",
                    "a pig walking around pen",
                    "a pig sitting without feeding",
                    "a pig sleeping peacefully",
                ],
            },
            "Drinking": {
                "positive": [
                    "a pig drinking water from nipple dispenser",
                    "a pig using a nipple drinker",
                    "a pig at the waterer taking a drink",
                    "a pig lapping water from water source",
                    "a pig accessing water for hydration",
                    "a pig with mouth on water nipple",
                ],
                "negative": [
                    "a pig eating from the feeder",
                    "a pig standing idle without drinking",
                    "a pig lying down resting",
                    "a pig walking without water access",
                    "a pig investigating environment",
                    "a pig sleeping",
                ],
            },
            "Standing": {
                "positive": [
                    "a pig standing upright on four legs",
                    "a pig in erect standing position",
                    "a pig standing still and alert",
                    "a pig maintaining upright posture",
                    "a pig standing stationary",
                    "a pig in vertical standing stance",
                ],
                "negative": [
                    "a pig lying on the floor",
                    "a pig sitting in dog-like position",
                    "a pig lying on its side",
                    "a pig resting on sternum",
                    "a pig in recumbent position",
                    "a pig sleeping on ground",
                ],
            },
            "Walking": {
                "positive": [
                    "a pig walking across the pen",
                    "a pig moving around enclosure",
                    "a pig in locomotion",
                    "a pig actively walking and moving",
                    "a pig showing movement behavior",
                    "a pig stepping forward in motion",
                ],
                "negative": [
                    "a pig standing still without movement",
                    "a pig lying motionless on floor",
                    "a pig sitting stationary",
                    "a pig sleeping without motion",
                    "a pig resting in fixed position",
                    "a pig stationary at feeder",
                ],
            },
            "Lying": {
                "positive": [
                    "a pig lying down on the floor",
                    "a pig in resting position on ground",
                    "a pig lying horizontally",
                    "a pig in recumbent position",
                    "a pig resting flat on surface",
                    "a pig lying down for rest",
                ],
                "negative": [
                    "a pig standing upright",
                    "a pig sitting in dog position",
                    "a pig walking around",
                    "a pig in vertical standing posture",
                    "a pig upright on four legs",
                    "a pig in seated position",
                ],
            },
            "Lateral lying": {
                "positive": [
                    "a pig lying completely on its side",
                    "a pig in lateral recumbent position",
                    "a pig lying sideways with legs extended",
                    "a pig lying on its side",
                    "a pig lying with full body contact on side",
                    "a pig in side-lying position",
                ],
                "negative": [
                    "a pig standing upright",
                    "a pig sitting upright",
                    "a pig lying on belly with head up",
                    "a pig walking around",
                    "a pig in sternal lying position",
                    "a pig upright on legs",
                ],
            },
            "Not lying": {
                "positive": [
                    "a pig not lying down",
                    "a pig in upright position not resting",
                    "a pig standing or sitting but not lying",
                    "a pig vertical not horizontal",
                    "a pig upright not recumbent",
                    "a pig active not lying down",
                ],
                "negative": [
                    "a pig lying down on the floor",
                    "a pig in recumbent position",
                    "a pig lying on its side",
                    "a pig resting flat on ground",
                    "a pig in sternal lying position",
                    "a pig in side-lying position",
                ],
            },
            "fight": {
                "positive": [
                    "pigs fighting aggressively with each other",
                    "pigs in aggressive physical conflict",
                    "pigs biting and head-knocking in fight",
                    "pigs pushing and charging aggressively",
                    "pigs engaged in agonistic interaction",
                    "pigs showing violent confrontation",
                ],
                "negative": [
                    "pigs in peaceful calm interaction",
                    "pigs coexisting without aggression",
                    "pigs showing non-aggressive behavior",
                    "pigs in friendly social contact",
                    "pigs interacting peacefully",
                    "single pig without social conflict",
                ],
            },
            "No fight": {
                "positive": [
                    "pigs showing no visible aggression",
                    "pigs in calm peaceful interaction",
                    "pigs coexisting without fighting",
                    "pigs displaying non-aggressive behavior",
                    "pigs in harmonious social behavior",
                    "pigs interacting without conflict",
                ],
                "negative": [
                    "pigs fighting aggressively with each other",
                    "pigs in violent conflict",
                    "pigs showing aggressive behavior",
                    "pigs biting and head-knocking in fight",
                    "pigs in hostile confrontation",
                    "pigs engaged in combat",
                ],
            },
            "Sleeping": {
                "positive": [
                    "a pig sleeping peacefully",
                    "a pig in deep sleep state",
                    "a pig resting with eyes closed",
                    "a pig sleeping comfortably",
                    "a pig in slumber",
                    "a pig dormant and sleeping",
                ],
                "negative": [
                    "a pig alert and awake",
                    "a pig actively walking",
                    "a pig eating at feeder",
                    "a pig drinking water",
                    "a pig standing alert",
                    "a pig actively exploring",
                ],
            },
            "Investigating": {
                "positive": [
                    "a pig investigating and exploring environment",
                    "a pig sniffing and nosing around",
                    "a pig rooting and exploring floor",
                    "a pig showing environmental exploration",
                    "a pig examining surroundings curiously",
                    "a pig nosing objects and surfaces",
                ],
                "negative": [
                    "a pig sleeping motionless",
                    "a pig eating at feeder",
                    "a pig drinking water",
                    "a pig lying still without exploration",
                    "a pig standing idle",
                    "a pig resting without investigation",
                ],
            },
            "Mounting": {
                "positive": [
                    "a pig mounting another pig",
                    "a pig riding on another pig's back",
                    "a pig climbing onto another pig",
                    "a pig showing mounting behavior for dominance",
                    "a pig displaying mounting social behavior",
                    "a pig on top of another pig",
                ],
                "negative": [
                    "pigs standing separately",
                    "a single pig without mounting",
                    "pigs lying down separately",
                    "pigs eating without mounting behavior",
                    "pigs in side-by-side position",
                    "pigs without physical dominance display",
                ],
            },
            "Active": {
                "positive": [
                    "a pig showing active energetic behavior",
                    "a pig walking and exploring actively",
                    "a pig displaying lively movement",
                    "a pig in active behavioral state",
                    "a pig showing vigorous activity",
                    "a pig alert and actively moving",
                ],
                "negative": [
                    "a pig sleeping peacefully",
                    "a pig lying motionless",
                    "a pig resting quietly",
                    "a pig standing still without activity",
                    "a pig in passive resting state",
                    "a pig dormant and inactive",
                ],
            },
            "Nose-to-nose": {
                "positive": [
                    "pigs touching noses in greeting",
                    "pigs with snouts touching nose-to-nose",
                    "pigs in nose contact interaction",
                    "pigs showing mutual nose sniffing",
                    "pigs engaged in social nosing",
                    "pigs with faces close nose contact",
                ],
                "negative": [
                    "pigs standing far apart",
                    "a single pig without social contact",
                    "pigs fighting aggressively",
                    "pigs showing side-by-side position",
                    "pigs without facial contact",
                    "pigs with backs to each other",
                ],
            },
            "Sitting": {
                "positive": [
                    "a pig sitting in dog-like posture",
                    "a pig in seated position with rear on ground",
                    "a pig sitting upright like dog",
                    "a pig in dog-sitting position",
                    "a pig seated with minimal movement",
                    "a pig in upright sitting stance",
                ],
                "negative": [
                    "a pig standing on four legs",
                    "a pig lying flat on floor",
                    "a pig lying on its side",
                    "a pig walking around",
                    "a pig in sternal lying position",
                    "a pig in standing upright posture",
                ],
            },
            "Sitting Drinking": {
                "positive": [
                    "a pig sitting while drinking water",
                    "a pig in seated position accessing water",
                    "a pig sitting at waterer drinking",
                    "a pig in dog-sit position drinking",
                    "a pig seated using nipple drinker",
                    "a pig sitting and hydrating",
                ],
                "negative": [
                    "a pig standing while drinking",
                    "a pig lying while drinking",
                    "a pig sitting while eating",
                    "a pig sitting without drinking",
                    "a pig walking to water",
                    "a pig standing at waterer",
                ],
            },
            "Sitting Feeding": {
                "positive": [
                    "a pig sitting while eating from feeder",
                    "a pig in seated position feeding",
                    "a pig sitting at trough eating",
                    "a pig in dog-sit position consuming feed",
                    "a pig seated while foraging at feeder",
                    "a pig sitting and eating",
                ],
                "negative": [
                    "a pig standing while eating",
                    "a pig lying while eating",
                    "a pig sitting while drinking",
                    "a pig sitting without eating",
                    "a pig walking to feeder",
                    "a pig standing at feeder",
                ],
            },
            "Sitting NF": {
                "positive": [
                    "a pig sitting without feeding activity",
                    "a pig in dog-sit position not eating",
                    "a pig seated without consuming food",
                    "a pig sitting without feeder access",
                    "a pig seated but not feeding",
                    "a pig in sitting posture without food",
                ],
                "negative": [
                    "a pig sitting while eating",
                    "a pig sitting while drinking",
                    "a pig standing without feeding",
                    "a pig lying without feeding",
                    "a pig walking around",
                    "a pig at feeder eating",
                ],
            },
            "Standing Drinking": {
                "positive": [
                    "a pig standing while drinking water",
                    "a pig upright at waterer drinking",
                    "a pig standing using nipple drinker",
                    "a pig in standing position accessing water",
                    "a pig upright while hydrating",
                    "a pig standing at water source",
                ],
                "negative": [
                    "a pig sitting while drinking",
                    "a pig lying while drinking",
                    "a pig standing while eating",
                    "a pig standing without drinking",
                    "a pig walking away from water",
                    "a pig lying at waterer",
                ],
            },
            "Standing Feeding": {
                "positive": [
                    "a pig standing while eating at feeder",
                    "a pig upright feeding at trough",
                    "a pig standing consuming feed",
                    "a pig in standing position eating",
                    "a pig upright foraging at feeder",
                    "a pig standing at feeding station",
                ],
                "negative": [
                    "a pig sitting while eating",
                    "a pig lying while eating",
                    "a pig standing while drinking",
                    "a pig standing without eating",
                    "a pig walking away from feeder",
                    "a pig lying at feeder",
                ],
            },
            "Standing NF": {
                "positive": [
                    "a pig standing without feeding activity",
                    "a pig upright not eating",
                    "a pig standing without consuming food",
                    "a pig in standing position without feeder",
                    "a pig upright but not feeding",
                    "a pig standing without food access",
                ],
                "negative": [
                    "a pig standing while eating",
                    "a pig standing while drinking",
                    "a pig sitting without feeding",
                    "a pig lying without feeding",
                    "a pig at feeder eating",
                    "a pig at waterer drinking",
                ],
            },
            "Sternal Lying Drinking": {
                "positive": [
                    "a pig lying on sternum while drinking",
                    "a pig in sternal position accessing water",
                    "a pig prone drinking from waterer",
                    "a pig chest-down using nipple drinker",
                    "a pig sternal lying while hydrating",
                    "a pig lying on chest drinking water",
                ],
                "negative": [
                    "a pig standing while drinking",
                    "a pig sitting while drinking",
                    "a pig side-lying while drinking",
                    "a pig sternal lying while eating",
                    "a pig sternal lying without drinking",
                    "a pig walking to waterer",
                ],
            },
            "Sternal Lying Feeding": {
                "positive": [
                    "a pig lying on sternum while eating",
                    "a pig in sternal position feeding",
                    "a pig prone eating from feeder",
                    "a pig chest-down consuming feed",
                    "a pig sternal lying while foraging",
                    "a pig lying on chest eating",
                ],
                "negative": [
                    "a pig standing while eating",
                    "a pig sitting while eating",
                    "a pig side-lying while eating",
                    "a pig sternal lying while drinking",
                    "a pig sternal lying without eating",
                    "a pig walking to feeder",
                ],
            },
            "Sternal Lying NF": {
                "positive": [
                    "a pig lying on sternum without feeding",
                    "a pig in sternal position not eating",
                    "a pig prone without consuming food",
                    "a pig chest-down without feeder access",
                    "a pig sternal lying but not feeding",
                    "a pig lying on chest without food",
                ],
                "negative": [
                    "a pig sternal lying while eating",
                    "a pig sternal lying while drinking",
                    "a pig standing without feeding",
                    "a pig sitting without feeding",
                    "a pig side-lying without feeding",
                    "a pig at feeder eating",
                ],
            },
        }

        self._norm_to_key: Dict[str, str] = {
            k.strip().lower(): k for k in self.label_descriptions
        }

    @staticmethod
    def _normalize(label: str) -> str:
        import re
        return re.sub(r'\s+', ' ', label.strip().lower())

    def resolve_label(self, label: str) -> Optional[str]:
        return self._norm_to_key.get(self._normalize(label), None)

    def get_positive_descriptions(self, label: str) -> List[str]:
        resolved = self.resolve_label(label) or label
        return self.label_descriptions.get(resolved, {}).get("positive", [f"a photo of {label}"])

    def get_negative_descriptions(self, label: str) -> List[str]:
        resolved = self.resolve_label(label) or label
        return self.label_descriptions.get(resolved, {}).get("negative", [f"not a photo of {label}"])

    def get_all_labels(self) -> List[str]:
        return list(self.label_descriptions.keys())
