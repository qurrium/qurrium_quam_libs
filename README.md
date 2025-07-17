# Qurrium Quam-Libs Crossroads 🚏 - The Converter Between Qurrium and Quam-Libs

## Classical Shadow

It returns single-shot result.

```py
import numpy as np
from quam_libs.experiments.classical_shadow import ClassicalShadow

job: ClassicalShadow

results: list[tuple[str, np.ndarray[int, np.dtype[int]]]] = job.result()
"""
list[tuple[str, np.ndarray[int, np.dtype[int]]]]:
    List of tuples containing the bitstring and the corresponding gate indices.
"""
```

```txt
# For examplem, a 3-qubit system
[({"010": 128}, [0, 1, 2]), ({"110": 128}, [2, 1, 0]), ...]
```

- bitstring order: "(last one)..(first one)"
- corresponding gate indices order: [(first one), ..., (last one)]

```py
ideal_results = job.ideal_result()
"""
list[tuple[dict[str, float], np.ndarray[int, np.dtype[int]]]]:
    List of tuples containing the probabilities and the corresponding gate indices.

"""
```

```txt
# For examplem, a 3-qubit system
[(
    {"010": 0.5,"110": 0.5}, [0, 1, 2]
), (
    {"110": 0.5, "010": 0.5}, [0, 1, 2]
), ...]
```

- bitstring order: "(last one)..(first one)"
- corresponding gate indices order: [(first one), ..., (last one)]

## Documentation

Builded by Sphinx.

```bash
pip install -r ./docs/requirements.txt
python -m sphinx docs docs/_build
```
