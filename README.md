# Qurrium Quam-Libs Crossroads 🚏 - The Converter Between Qurrium and Quam-Libs

## Classical Shadow

It returns single-shot result.

```py
from quam_libs.experiments.classical_shadow import ClassicalShadow

job: ClassicalShadow

results: list[tuple[str, np.ndarray[int, np.dtype[int]]]] = job.result()
"""
list[tuple[str, np.ndarray[int, np.dtype[int]]]]:
    List of tuples containing the bitstring and the corresponding gate indices.

.. code-block:: txt
    # For examplem, a 3-qubit system
    [("010", [0, 1, 2]), ("110", [2, 1, 0]), ...]
"""
```

```py
ideal_results = job.ideal_result()
"""
list[tuple[dict[str, float], np.ndarray[int, np.dtype[int]]]]:
    List of tuples containing the probabilities and the corresponding gate indices.

.. code-block:: txt
    # For examplem, a 3-qubit system
    [(
        {"010": 0.5,"110": 0.5}, [0, 1, 2]
    ), (
        {"110": 0.5, "010": 0.5}, [0, 1, 2]
    ), ...]
"""
```
