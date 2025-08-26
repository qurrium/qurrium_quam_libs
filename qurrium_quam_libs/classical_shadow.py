"""Qurrium Quam-Libs Crossroads - Classical Shadow Conversion Module
(:mod:`qurrium_quam_libs.classical_shadow`)

This module provides functions to transform
the output of Qurrium and Quam_Libs on classical shadow to each other.
"""

from typing import Literal, Any
from collections.abc import Sequence
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qurry.qurrent.classical_shadow import ShadowUnveilExperiment
from qurry.qurrent.classical_shadow.utils import circuit_method_core as shadow_circuit_maker
from qurry.qurrium.experiment.utils import exp_id_process
from qurry.qurrium.utils import qasm_dumps
from qurry.tools import current_time, DatetimeDict

from .utils import QURRIUM_VERSION, check_qua_libs_single_shots_results, check_qua_libs_results


# qurrium to qua_libs transformation
def check_classical_shadow_exp(
    classical_shadow_exp: ShadowUnveilExperiment, is_single_shot: bool = False
) -> tuple[dict[int, dict[int, int]], dict[int, int]]:
    """Check if the classical shadow experiment is valid for conversion,
    then return its random basis and registers mapping.

    Args:
        classical_shadow_exp (ShadowUnveilExperiment):
            The Qurry experiment to check.
        is_single_shot (bool): If True, checks for single-shot experiments.

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
    measured_qubits = set(classical_shadow_exp.args.qubits_measured)
    unitary_located = set(classical_shadow_exp.args.unitary_located)
    missing_qubits = measured_qubits - unitary_located
    if missing_qubits:
        raise ValueError(
            "All measured qubits must be part of the unitary located. "
            + f"Missing qubits: {sorted(missing_qubits)}"
        )
    if classical_shadow_exp.commons.shots != 1 and is_single_shot:
        raise ValueError("The number of shots must be 1 for QuaLibs conversion.")

    if (0, 13, 0) < QURRIUM_VERSION:
        if classical_shadow_exp.args.random_basis is None:
            raise ValueError("The experiment must have 'random_basis' in args.")
        random_basis = classical_shadow_exp.args.random_basis
    else:
        if "random_unitary_ids" not in classical_shadow_exp.beforewards.side_product:
            raise ValueError("The experiment must have 'random_unitary_ids' in side products.")
        random_basis: dict[int, dict[int, int]] = classical_shadow_exp.beforewards.side_product[
            "random_unitary_ids"
        ]
    return random_basis, classical_shadow_exp.args.registers_mapping


def get_gate_indices(
    single_random_basis: dict[int, int], registers_mapping: dict[int, int]
) -> list[int]:
    """Get the gate indices for the QuaLibs format.

    Args:
        single_random_basis (dict[int, int]):
            Mapping of qubit indices to random basis.
        registers_mapping (dict[int, int]):
            Mapping of qubit indices and classical registers.
    Returns:
        list[int]: A list of gate indices corresponding to the qubits.
    """
    return [single_random_basis[qi] for qi in registers_mapping.keys()]


def single_shots_processing(
    idx: int,
    single_shot_counts: dict[str, int],
    random_basis: dict[int, int],
    registers_mapping: dict[int, int],
) -> tuple[str, list[int]]:
    """Single shot processing for QuaLibs format.

    Args:
        idx (int):
            The index of the single shot counts and random unitary.
        single_shot_counts (dict[str, int]):
            The counts of single-shot results.
        random_basis (dict[int, int]):
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
        get_gate_indices(random_basis, registers_mapping),
    )


def qurrium_single_shot_to_qua_libs_single_shot_result(
    classical_shadow_exp: ShadowUnveilExperiment,
) -> list[tuple[str, list[int]]]:
    """Convert a Qurrium single-shot experiment to QuaLibs format single-shot result.

    Args:
        classical_shadow_exp (ShadowUnveil): The Qurry experiment to convert.

    Returns:
        list[tuple[str, list[int]]]:
            List of tuples containing the bitstring and the corresponding gate indices.
    """
    random_basis, registers_mapping = check_classical_shadow_exp(
        classical_shadow_exp, is_single_shot=True
    )

    result = [
        single_shots_processing(idx, single_counts, single_random_unitary_id, registers_mapping)
        for (idx, single_random_unitary_id), single_counts in zip(
            random_basis.items(), classical_shadow_exp.afterwards.counts
        )
    ]
    return result


def multiple_shots_processing(
    single_counts: dict[str, int],
    single_random_basis: dict[int, int],
    registers_mapping: dict[int, int],
) -> tuple[dict[str, int], list[int]]:
    """Multiple shot processing for QuaLibs format.

    Args:
        single_counts (dict[str, int]):
            The counts of single-shot results.
        single_random_basis (dict[int, int]):
            Mapping of qubit indices to random basis.
        registers_mapping (dict[int, int]):
            Mapping of qubit indices and classical registers.

    Returns:
        tuple[dict[str, int], list[int]]:
            A tuple containing the bitstring and the corresponding gate indices.
    """

    return (
        {k[: len(registers_mapping)]: v for k, v in single_counts.items()},
        get_gate_indices(single_random_basis, registers_mapping),
    )


def qurrium_to_qua_libs_result(
    classical_shadow_exp: ShadowUnveilExperiment,
) -> list[tuple[dict[str, int], list[int]]]:
    """Convert a Qurrium experiment to QuaLibs format result.

    Args:
        classical_shadow_exp (ShadowUnveil): The Qurrium experiment to convert.

    Returns:
        list[tuple[str, list[int]]]:
            List of tuples containing the bitstring and the corresponding gate indices.
    """
    random_basis, registers_mapping = check_classical_shadow_exp(classical_shadow_exp)

    result = [
        multiple_shots_processing(single_counts, single_random_unitary_id, registers_mapping)
        for (idx, single_random_unitary_id), single_counts in zip(
            random_basis.items(), classical_shadow_exp.afterwards.counts
        )
    ]
    return result


def processing_to_ideal_result(
    idx: int,
    circuit: QuantumCircuit,
    random_basis: dict[int, int],
    registers_mapping: dict[int, int],
) -> tuple[dict[str, float], list[int]]:
    """Single shot processing for QuaLibs format.

    Args:
        idx (int):
            The index of the single shot counts and random unitary.
        circuit (QuantumCircuit):
            The quantum circuit to process.
        random_basis (dict[int, int]):
            Mapping of qubit indices to random basis.
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

    return probs, get_gate_indices(random_basis, registers_mapping)


def qurrium_to_qua_libs_ideal_result(
    classical_shadow_exp: ShadowUnveilExperiment,
) -> list[tuple[dict[str, float], list[int]]]:
    """Convert a Qurrium experiment to QuaLibs format ideal result.

    Args:
        classical_shadow_exp (ShadowUnveil): The Qurry experiment to convert.

    Returns:
        list[tuple[dict[str, float], list[int]]]:
            List of tuples containing the probabilities and the corresponding gate indices.
    """
    random_basis, registers_mapping = check_classical_shadow_exp(classical_shadow_exp)

    ideal_result = [
        processing_to_ideal_result(idx, circuit, single_random_unitary_id, registers_mapping)
        for (idx, single_random_unitary_id), circuit in zip(
            random_basis.items(),
            classical_shadow_exp.beforewards.circuit,
        )
    ]
    return ideal_result


# qua_libs to qurrium transformation
def qua_libs_single_shot_result_to_qurrium_single_shot(
    result: Sequence[tuple[str, Sequence[int]]],
    exp_name: str = "experiment.qua_libs",
    tags: tuple[str, ...] | None = None,
    save_location: str | None = None,
    backend_name: str | None = None,
) -> ShadowUnveilExperiment:
    """Convert QuaLibs single-shot result to Qurrium single-shot experiment format.
    We assume that the number of qubits is equal to the number of classical registers.
    And their mapping is identity.

    Args:
        result (Sequence[tuple[str, Sequence[int]]]): The QuaLibs result to convert.
        exp_name (str): The name of the experiment.
        tags (tuple[str, ...] | None): Tags for the experiment.
        save_location (str | None): The location to save the experiment.
        backend_name (str | None): The name of the backend used for the experiment.

    Returns:
        ShadowUnveilExperiment: The converted Qurrium single-shot experiment.
    """
    num_classical_register, num_random_basis = check_qua_libs_single_shots_results(result)

    args = {
        "exp_name": exp_name,
        "qubits_measured": list(range(num_classical_register)),
        "registers_mapping": {qi: qi for qi in range(num_random_basis)},
        "actual_num_qubits": num_classical_register,
        "unitary_located": list(range(num_random_basis)),
    }
    if QURRIUM_VERSION > (0, 13, 0):
        args["snapshots"] = len(result)
        args["random_basis"] = {}
    else:
        args["times"] = len(result)

    commons = {
        "exp_id": exp_id_process(None),
        "target_keys": [],
        "shots": 1,
        "backend": "qua_libs_transformed" + (f"-{backend_name}" if backend_name else ""),
        "run_args": {},
        "transpile_args": {},
        "tags": tags,
        "save_location": Path(save_location) if save_location else Path("./"),
        "serial": None,
        "summoner_id": None,
        "summoner_name": None,
        "datetimes": {"transform-from-qua_libs": current_time()},
    }
    if QURRIUM_VERSION < (0, 13, 0):
        commons["filename"] = ""
        commons["files"] = {}

    outfields = {
        "denoted": "This is a QuaLibs result converted to Qurrium single-shot format.",
    }

    classical_shadow_exp = ShadowUnveilExperiment(
        arguments=args, commonparams=commons, outfields=outfields
    )

    counts = [{bitstring: 1} for bitstring, gate_indices in result]
    classical_shadow_exp.afterwards.counts.extend(counts)

    random_basis = {
        idx: dict(enumerate(gate_indices)) for idx, (bitstring, gate_indices) in enumerate(result)
    }
    if QURRIUM_VERSION > (0, 13, 0):
        assert classical_shadow_exp.args.random_basis is not None, "Random basis must be set."
        classical_shadow_exp.args.random_basis.update(random_basis)
    else:
        classical_shadow_exp.beforewards.side_product["random_unitary_ids"] = random_basis

    return classical_shadow_exp


def qua_libs_result_to_qurrium(
    result: Sequence[tuple[dict[str, int], Sequence[int]]],
    shots_per_snapshot: int,
    exp_name: str = "experiment.qua_libs",
    target_circuit: QuantumCircuit | None = None,
    backend_name: str | None = None,
    tags: tuple[str, ...] | None = None,
    datetimes: DatetimeDict | dict[str, str] | None = None,
    outfields: dict[str, Any] | None = None,
    # multimanager
    serial: int | None = None,
    summoner_id: str | None = None,
    summoner_name: str | None = None,
    # process tool
    qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
    save_location: str | None = None,
) -> ShadowUnveilExperiment:
    """Convert QuaLibs result to Qurrium experiment format.
    We assume that the number of qubits is equal to the number of classical registers.
    And their mapping is identity.

    Args:
        result (Sequence[tuple[dict[str, int], Sequence[int]]]): The QuaLibs result to convert.
        shots_per_snapshot (int): The number of shots per snapshot in QuaLibs.
        exp_name (str): The name of the experiment.
        target_circuit (QuantumCircuit | None): The target circuit for the experiment.
        backend_name (str | None): The name of the backend used for the experiment.
        tags (tuple[str, ...] | None): Tags for the experiment.
        datetimes (DatetimeDict | dict[str, str] | None):
            Datetime information for the experiment.
        outfields (dict[str, Any] | None): Additional data to include.

        serial (int | None):
            Serial number for the experiment in :class:`~qurry.qurrium.multimanager.MultiManager`.
        summoner_id (str | None):
            ID of the summoner for :class:`~qurry.qurrium.multimanager.MultiManager`.
        summoner_name (str | None):
            Name of the summoner for :class:`~qurry.qurrium.multimanager.MultiManager`.

        qasm_version (Literal['qasm2', 'qasm3']):
            The OpenQASM version to use for the circuit. Defaults to 'qasm3'.
        save_location (str | None): The location to save the experiment.

    Returns:
        ShadowUnveilExperiment: The converted Qurrium experiment.
    """
    if target_circuit is not None and not isinstance(target_circuit, QuantumCircuit):
        raise TypeError("The target_circuit must be a QuantumCircuit instance.")
    num_classical_register, num_random_basis = check_qua_libs_results(result)

    args = {
        "exp_name": exp_name,
        "qubits_measured": list(range(num_classical_register)),
        "registers_mapping": {qi: qi for qi in range(num_random_basis)},
        "actual_num_qubits": num_classical_register,
        "unitary_located": list(range(num_random_basis)),
    }
    if QURRIUM_VERSION > (0, 13, 0):
        args["snapshots"] = len(result)
        args["random_basis"] = {}
    else:
        args["times"] = len(result)

    commons = {
        "exp_id": exp_id_process(None),
        "target_keys": [],
        "shots": shots_per_snapshot,
        "backend": "qua_libs_transformed" + (f"-{backend_name}" if backend_name else ""),
        "run_args": {},
        "transpile_args": {},
        "tags": tags,
        "save_location": Path(save_location) if save_location else Path("./"),
        "serial": serial,
        "summoner_id": summoner_id,
        "summoner_name": summoner_name,
        "datetimes": datetimes,
    }
    if QURRIUM_VERSION < (0, 13, 0):
        commons["filename"] = ""
        commons["files"] = {}

    outfields_inner = {} if outfields is None else outfields.copy()
    outfields_inner["denoted"] = "This is a QuaLibs result converted to Qurrium format."

    classical_shadow_exp = ShadowUnveilExperiment(
        arguments=args, commonparams=commons, outfields=outfields_inner
    )
    classical_shadow_exp.commons.datetimes.add_only("transform-from-qua_libs")

    counts = [single_counts for single_counts, gate_indices in result]
    counts_sum_neq = [
        idx
        for idx, single_counts in enumerate(counts)
        if sum(single_counts.values()) != shots_per_snapshot
    ]
    if counts_sum_neq:
        raise ValueError(
            "The sum of counts values must be equal "
            f"to shots_per_snapshot ({shots_per_snapshot}). "
            f"Found in following index: {counts_sum_neq}."
        )
    classical_shadow_exp.afterwards.counts.extend(counts)

    random_basis = {
        idx: dict(enumerate(gate_indices))
        for idx, (single_counts, gate_indices) in enumerate(result)
    }
    if QURRIUM_VERSION > (0, 13, 0):
        assert classical_shadow_exp.args.random_basis is not None, "Random basis must be set."
        classical_shadow_exp.args.random_basis.update(random_basis)
    else:
        classical_shadow_exp.beforewards.side_product["random_unitary_ids"] = random_basis

    if target_circuit is not None:
        classical_shadow_exp.beforewards.target.append((exp_name, target_circuit))
        classical_shadow_exp.beforewards.target_qasm.append(
            (exp_name, qasm_dumps(target_circuit, qasm_version))
        )

        classical_shadow_exp.beforewards.circuit.clear()
        classical_shadow_exp.beforewards.circuit_qasm.clear()
        classical_shadow_exp.beforewards.circuit.extend(
            [
                shadow_circuit_maker(
                    idx,
                    target_circuit,
                    "",
                    classical_shadow_exp.args.exp_name,
                    args["registers_mapping"],
                    gate_indices,
                )
                for idx, gate_indices in random_basis.items()
            ]
        )
        classical_shadow_exp.beforewards.circuit_qasm.extend(
            [qasm_dumps(q, qasm_version) for q in classical_shadow_exp.beforewards.circuit]
        )

    return classical_shadow_exp
