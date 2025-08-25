"""Qurrium Quam-Libs Crossroads - Utilities (:mod:`qurrium_quam_libs.utils`)"""

from collections.abc import Sequence

from qurry import __version__
from qurry.tools import DatetimeDict


def get_version_info() -> tuple[int, int, int]:
    """Get the version information of the Qurrium package.

    Returns:
        tuple[int, int, int]: The major, minor, and patch version numbers.
    """
    version_parts = __version__.split(".")[:3]
    version_parts += ["0"] * (3 - len(version_parts))
    return tuple(map(int, version_parts))  # type: ignore


QURRIUM_VERSION = get_version_info()
"""The current version of Qurrium."""


def check_tags(tags: tuple[str, ...] | None) -> tuple[str, ...]:
    """Check tags and return formatted tags.

    Args:
        tags (tuple[str, ...] | None): Tags for the experiment.

    Returns:
        tuple[str, ...]: Formatted tags for the experiment.
    """
    if tags is None:
        tags = ()
    elif not isinstance(tags, tuple):
        raise TypeError("Tags must be a tuple of strings.")

    if any(not isinstance(tag, str) for tag in tags):
        raise TypeError("Tags must be a tuple of strings, not a tuple of strings and other types.")

    return tags


def check_datatimes(datetimes: DatetimeDict | dict[str, str] | None) -> DatetimeDict:
    """Check and format the datetimes dictionary.

    Args:
        datetimes (DatetimeDict | dict[str, str] | None): The datetimes dictionary.

    Returns:
        DatetimeDict: The formatted datetimes dictionary.
    """
    if datetimes is None:
        datetimes = DatetimeDict()
    elif not isinstance(datetimes, (DatetimeDict, dict)):
        raise TypeError("Datetimes must be a DatetimeDict or a dictionary.")
    for key, value in datetimes.items():
        if not isinstance(value, str):
            raise TypeError(
                f"All values in datetimes must be strings. Found {value} for key {key}."
            )
    datetimes = DatetimeDict(datetimes)
    datetimes.add_only("transform-from-qua_libs")

    return datetimes


def check_qua_libs_single_shots_results(
    result: Sequence[tuple[str, Sequence[int]]],
) -> tuple[int, int]:
    """Check the single-shot results from Qua-Libs and
    return the number of classical registers and random bases.

    Args:
        result (Sequence[tuple[str, Sequence[int]]]): The results from Qua-Libs.

    Returns:
        tuple[int, int]: The number of classical registers and random bases.
    """
    if not isinstance(result, Sequence):
        raise TypeError("The result must be a sequence of tuples.")
    if len(result) < 1:
        raise ValueError("The result must contain at least one tuple.")

    invalid_result = [
        idx
        for idx, (bitstring, single_basis) in enumerate(result)
        if not (isinstance(bitstring, str) and isinstance(single_basis, Sequence))
    ]
    if invalid_result:
        raise TypeError(
            f"The result must be a sequence of tuples (str, Sequence[int]). "
            f"Invalid entries at indices: {invalid_result}."
        )

    sample_bitstring, sample_unitary_ids = result[0]
    num_classical_register = len(sample_bitstring)
    num_random_basis = len(sample_unitary_ids)

    return num_classical_register, num_random_basis


def validate_single_counts(single_counts) -> bool:
    """Validate the single counts dictionary.

    Args:
        single_counts (dict[str, int]): The single counts dictionary to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(single_counts, dict):
        return False
    if not all(isinstance(k, str) and isinstance(v, int) for k, v in single_counts.items()):
        return False
    return True


def check_qua_libs_results(
    result: Sequence[tuple[dict[str, int], Sequence[int]]],
) -> tuple[int, int]:
    """Check the results from Qua-Libs and
    return the number of classical registers and random bases.

    Args:
        result (Sequence[tuple[dict[str, int], Sequence[int]]]): The results from Qua-Libs.

    Returns:
        tuple[int, int]: The number of classical registers and random bases.
    """
    if not isinstance(result, Sequence):
        raise TypeError("The result must be a sequence of tuples.")
    if len(result) < 1:
        raise ValueError("The result must contain at least one tuple.")

    invalid_result = [
        idx
        for idx, (single_counts, single_basis) in enumerate(result)
        if not (validate_single_counts(single_counts) and isinstance(single_basis, Sequence))
    ]
    if invalid_result:
        raise TypeError(
            f"The result must be a sequence of tuples (dict[str, int], Sequence[int]). "
            f"Invalid entries at indices: {invalid_result}."
        )

    sample_bitstring, sample_unitary_ids = result[0]
    num_classical_register = len(sample_bitstring)
    num_random_basis = len(sample_unitary_ids)

    return num_classical_register, num_random_basis
