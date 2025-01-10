import jax
import math
from enum import Enum
from typing import Dict, Any, Callable


class SchedulerEnum(Enum):
    LinearLR = 1
    CosineAnnealingLR = 2

def linear_schedule(count: int, config: Dict[str, Any]) -> float:
    num_mini_batch = config.get("NUM_MINIBATCHES", 1)
    update_epoches = config.get("UPDATE_EPOCHS", 1)
    num_updates = config.get("NUM_UPDATES", 1)
    learning_rate = config.get("LR", 1.0)

    frac = 1.0 - (count // num_mini_batch * update_epoches) / num_updates

    return learning_rate * frac

def cosine_annealing_schedule(count: int, config: Dict[str, Any]) -> float:
    """
    Cosine Annealing learning rate schedule.

    :param count: Current step or epoch.
    :param config: Configuration dictionary containing:
                   - LR: Initial learning rate (eta_max).
                   - MIN_LR: Minimum learning rate (eta_min).
                   - T_MAX: Maximum number of steps or epochs for one cycle.
    :return: Current learning rate.
    """
    initial_lr = config.get("LR", 1.0)  # Initial learning rate (eta_max)
    min_lr = config.get("MIN_LR", 0.0)  # Minimum learning rate (eta_min)
    t_max = config.get("T_MAX", 100)    # Maximum number of steps or epochs

    # Calculate the current learning rate using the cosine formula
    cosine_value = math.cos((count % t_max) / t_max * math.pi)
    current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cosine_value)

    return current_lr

# create a scheduler mapping dict
scheduler_mapping = {
    SchedulerEnum.LinearLR: linear_schedule,
    SchedulerEnum.CosineAnnealingLR: cosine_annealing_schedule,
}

def get_scheduler_from_str(scheduler_str: str) -> SchedulerEnum:
    """
    Converts a string to the corresponding SchedulerEnum value.

    :param scheduler_str: The string representation of the scheduler (e.g., "LinearLR").
    :return: The corresponding SchedulerEnum value.
    :raises ValueError: If the string does not match any SchedulerEnum member.
    """
    # Check if the string matches any SchedulerEnum member
    if scheduler_str in SchedulerEnum.__members__:
        return SchedulerEnum[scheduler_str]
    else:
        # Raise an error if the string does not match any enum member
        raise ValueError(f"Unknown scheduler name: {scheduler_str}. "
                         f"Available options are: {list(SchedulerEnum.__members__.keys())}")

def get_scheduler_handler(config: Dict[str, Any]) -> Callable[[int], float]:
    """
    generate scheduler handler based on scheduler name

    :param scheduler_name: name of sceduler, type be SchedulerEnum
    :param config: config dict, contain all parameters of scheduler
    :return: return a scheduler function, this function contains count and return a float number
    """
    scheduler_name = config.get("SCHEDULER", "InvalidScheduer")
    if scheduler_name in scheduler_mapping:
        return lambda count: scheduler_mapping[scheduler_name](count, config)
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")
