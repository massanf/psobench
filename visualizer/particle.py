from typing import Dict, Union, List


class Particle:
    def __init__(self, datum: Dict[str, Union[float, List[float]]]) -> None:
        if isinstance(datum["pos"], list) and all(isinstance(i, float)
                                                  for i in datum["pos"]):
            self.pos: List[float] = datum["pos"]
        else:
            val = datum["pos"]
            raise ValueError(f"Wrong value type for `pos`: {val}")

        if isinstance(datum["vel"], list) and all(isinstance(i, float)
                                                  for i in datum["vel"]):
            self.vel: List[float] = datum["vel"]
        else:
            raise ValueError("Wrong value type for `vel`.")

        if isinstance(datum["fitness"], float):
            self.fitness: float = datum["fitness"]
        else:
            raise ValueError("Wrong value type for `fitness`.")
