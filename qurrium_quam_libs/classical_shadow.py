"""Qurrium-Qua_Libs Classical Shadow Output Transformation

This module provides functions to transform
the output of Qurrium and Qua_Libs on classical shadow to each other.
"""

from typing import Optional
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qurry.qurrent.classical_shadow import ShadowUnveilExperiment
from qurry.qurrium.experiment.utils import exp_id_process
from qurry.tools import current_time


# qurrium to qua_libs transformation
def check_classical_shadow_exp(
    classical_shadow_exp: ShadowUnveilExperiment,
) -> tuple[dict[int, dict[int, int]], dict[int, int]]:
    """
    Check if the classical shadow experiment is valid for conversion.

    Args:
        classical_shadow_exp (ShadowUnveilExperiment):
            The Qurry experiment to check.

    Raises:
        TypeError: If the input is not a ShadowUnveilExperiment.
        ValueError: If required attributes are missing or invalid.

    Returns:
        tuple[dict[int, dict[int, int]], dict[int, int]]:
            random_unitary_ids mapping from the experiment and registers mapping.
    """
    if not isinstance(classical_shadow_exp, ShadowUnveilExperiment):
        raise TypeError(
            "The input must be an instance of ShadowUnveilExperiment "
            "from qurry.qurrent.classical_shadow."
        )
    if classical_shadow_exp.args.unitary_located is None:
        raise ValueError("The unitary located must be specified in the experiment.")
    if classical_shadow_exp.args.qubits_measured is None:
        raise ValueError("The qubits measured must be specified in the experiment.")
    if classical_shadow_exp.args.registers_mapping is None:
        raise ValueError("The registers mapping must be specified in the experiment.")
    if any(
        qi not in classical_shadow_exp.args.unitary_located
        for qi in classical_shadow_exp.args.qubits_measured
    ):
        raise ValueError("All measured qubits must be part of the unitary located.")
    if classical_shadow_exp.commons.shots != 1:
        raise ValueError("The number of shots must be 1 for QuaLibs conversion.")
    if "random_unitary_ids" not in classical_shadow_exp.beforewards.side_product:
        raise ValueError("The experiment must have 'random_unitary_ids' in side products.")

    random_unitary_ids: dict[int, dict[int, int]] = classical_shadow_exp.beforewards.side_product[
        "random_unitary_ids"
    ]
    return random_unitary_ids, classical_shadow_exp.args.registers_mapping


def get_gate_indices(
    random_unitary_ids: dict[int, int],
    registers_mapping: dict[int, int],
) -> list[int]:
    """
    Get the gate indices for the QuaLibs format.

    Args:
        random_unitary_ids (dict[int, int]):
            Mapping of qubit indices to random unitary.
        registers_mapping (dict[int, int]):
            Mapping of qubit indices and classical registers.
    Returns:
        list[int]: A list of gate indices corresponding to the qubits.
    """
    return [random_unitary_ids[qi] for qi in registers_mapping.keys()]


def single_shots_processing(
    idx: int,
    single_shot_counts: dict[str, int],
    random_unitary_ids: dict[int, int],
    registers_mapping: dict[int, int],
) -> tuple[str, list[int]]:
    """Single shot processing for QuaLibs format.
    Args:
        idx (int):
            The index of the single shot counts and random unitary.
        single_shot_counts (dict[str, int]):
            The counts of single-shot results.
        random_unitary_ids (dict[int, int]):
            Mapping of qubit indices to random unitary.
        registers_mapping (dict[int, int]):
            Mapping of qubit indices and classical registers.

    Returns:
        tuple[str, list[int]]:
            A tuple containing the bitstring and the corresponding gate indices.
    """

    if len(single_shot_counts) != 1:
        raise ValueError(
            "There should be exactly one key in single_shot_counts."
            + f" Found: {len(single_shot_counts)} keys in index {idx}."
        )
    only_bitstring = list(single_shot_counts.keys())[0]
    return (
        only_bitstring[: len(registers_mapping)],
        get_gate_indices(random_unitary_ids, registers_mapping),
    )


def qurrium_single_shot_to_qua_libs_result(
    classical_shadow_exp: ShadowUnveilExperiment,
) -> list[tuple[str, list[int]]]:
    """
    Convert a Qurrium single-shot experiment to QuaLibs format result.

    Args:
        qurry_exp (ShadowUnveil): The Qurry experiment to convert.

    Returns:
        list[tuple[str, list[int]]]:
            List of tuples containing the bitstring and the corresponding gate indices.
    """
    random_unitary_ids, registers_mapping = check_classical_shadow_exp(classical_shadow_exp)

    result = [
        single_shots_processing(idx, single_counts, single_random_unitary_id, registers_mapping)
        for (idx, single_random_unitary_id), single_counts in zip(
            random_unitary_ids.items(), classical_shadow_exp.afterwards.counts
        )
    ]
    return result


def single_shots_processing_ideal(
    idx: int,
    circuit: QuantumCircuit,
    random_unitary_ids: dict[int, int],
    registers_mapping: dict[int, int],
) -> tuple[dict[str, float], list[int]]:
    """Single shot processing for QuaLibs format.
    Args:
        idx (int):
            The index of the single shot counts and random unitary.
        circuit (QuantumCircuit):
            The quantum circuit to process.
        random_unitary_ids (dict[int, int]):
            Mapping of qubit indices to random unitary.
        registers_mapping (dict[int, int]):
            Mapping of qubit indices and classical registers.

    Returns:
        tuple[dict[str, float], list[int]]:
            A tuple containing the bitstring and the corresponding gate indices.
    """
    no_measiure_circ = circuit.remove_final_measurements(inplace=False)
    assert no_measiure_circ is not None, f"Circuit must have measurements removed. Index: {idx}"
    state = Statevector(no_measiure_circ)
    probs: dict[str, float] = state.probabilities_dict()

    return probs, get_gate_indices(random_unitary_ids, registers_mapping)


def qurrium_single_shot_to_qua_libs_ideal_result(
    classical_shadow_exp: ShadowUnveilExperiment,
) -> list[tuple[dict[str, float], list[int]]]:
    """
    Convert a Qurrium single-shot experiment to QuaLibs format ideal result.

    Args:
        classical_shadow_exp (ShadowUnveil): The Qurry experiment to convert.

    Returns:
        list[tuple[dict[str, float], list[int]]]:
            List of tuples containing the probabilities and the corresponding gate indices.
    """
    random_unitary_ids, registers_mapping = check_classical_shadow_exp(classical_shadow_exp)

    ideal_result = [
        single_shots_processing_ideal(idx, circuit, single_random_unitary_id, registers_mapping)
        for (idx, single_random_unitary_id), circuit in zip(
            random_unitary_ids.items(),
            classical_shadow_exp.beforewards.circuit,
        )
    ]
    return ideal_result


# qua_libs to qurrium transformation
def qua_libs_result_to_qurrium_single_shot(
    result: list[tuple[str, list[int]]],
    exp_name: str = "experiment.qua_libs",
    tags: Optional[tuple[str, ...]] = None,
    save_location: Optional[str] = None,
) -> ShadowUnveilExperiment:
    """
    Convert QuaLibs result to Qurrium single-shot experiment format.
    We assume that the number of qubits is equal to the number of classical registers.
    And their mapping is identity.

    Args:
        result (list[tuple[str, list[int]]]): The QuaLibs result to convert.
        exp_name (str): The name of the experiment.
        tags (Optional[tuple[str, ...]]): Tags for the experiment.
        save_location (Optional[str]): The location to save the experiment.

    Returns:
        ShadowUnveilExperiment: The converted Qurrium single-shot experiment.
    """
    if tags is None:
        tags = ()
    elif not isinstance(tags, tuple):
        raise TypeError("Tags must be a tuple of strings.")

    sample_bitstring, sample_unitary_ids = result[0]
    if not isinstance(sample_bitstring, str):
        raise ValueError("The first element of the result must be a bitstring (str).")

    num_classical_register = len(sample_bitstring)
    num_random_unitary = len(sample_unitary_ids)

    args = {
        "exp_name": exp_name,
        "times": len(result),
        "qubits_measured": list(range(len(sample_bitstring))),
        "registers_mapping": {qi: qi for qi in range(num_random_unitary)},
        "actual_num_qubits": num_classical_register,
        "unitary_located": list(range(num_random_unitary)),
    }
    commons = {
        "exp_id": exp_id_process(None),
        "target_keys": [],
        "shots": 1,
        "backend": "qua_libs_transformed",
        "run_args": {},
        "transpile_args": {},
        "tags": tags,
        "save_location": Path(save_location) if save_location else Path("./"),
        "serial": None,
        "summoner_id": None,
        "summoner_name": None,
        "datetimes": {
            "transform-from-qua_libs": current_time(),
        },
    }
    outfields = {
        "denoted": "This is a QuaLibs result converted to Qurrium single-shot format.",
    }

    classical_shadow_exp = ShadowUnveilExperiment(
        arguments=args, commonparams=commons, outfields=outfields
    )

    classical_shadow_exp.beforewards.side_product["random_unitary_ids"] = {}
    for idx, (bitstring, gate_indices) in enumerate(result):
        classical_shadow_exp.afterwards.counts.append({bitstring: 1})
        classical_shadow_exp.beforewards.side_product["random_unitary_ids"][idx] = {
            qi: gate_idx for qi, gate_idx in enumerate(gate_indices)
        }

    return classical_shadow_exp
