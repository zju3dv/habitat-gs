from typing import List

from .mesh_human_actor import MeshHumanActor


class MeshHumanManager:
    def __init__(self, actors: List[MeshHumanActor]):
        self.actors = list(actors)

    def meshes_at(self, sim_time: float) -> List[dict]:
        return [actor.mesh_at(sim_time) for actor in self.actors]

    def capsules_at(self, sim_time: float) -> List[dict]:
        return [actor.capsule_at(sim_time) for actor in self.actors]
