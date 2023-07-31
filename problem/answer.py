import sys
from typing import Any

import numpy as np
from openfermion.transforms import jordan_wigner, bravyi_kitaev
from openfermion.utils import load_operator

from quri_parts.circuit import QuantumCircuit
from quri_parts.algo.ansatz import HardwareEfficientReal, HardwareEfficient
# from quri_parts.algo.optimizer import Adam, OptimizerStatus
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)

from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState, GeneralCircuitQuantumState
from quri_parts.openfermion.operator import operator_from_openfermion_op
from quri_parts.core.operator import PAULI_IDENTITY

sys.path.append("../")
from utils.challenge_2023 import ChallengeSampling, TimeExceededError


#####################################
############## SPSA #################
#####################################


from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from quri_parts.core.utils.array import readonly_array

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import numpy as np  # noqa: F401
    import numpy.typing as npt  # noqa: F401

#: Represents a parameter vector subject to optimization.
#: (A gradient vector is also represented as Params.)
Params: TypeAlias = "npt.NDArray[np.float_]"

#: Cost function for optimization.
CostFunction: TypeAlias = Callable[[Params], float]

#: Gradient function for optimization.
GradientFunction: TypeAlias = Callable[[Params], Params]


class OptimizerStatus(Enum):
    """Status of optimization."""

    #: No error, not converged yet.
    SUCCESS = auto()
    #: The optimization failed and cannot be continued.
    FAILED = auto()
    #: The optimization converged.
    CONVERGED = auto()


@dataclass(frozen=True)
class OptimizerState:
    """An immutable (frozen) dataclass representing an optimizer state."""

    #: Current parameter values.
    params: Params
    #: Current value of the cost function.
    cost: float = 0.0
    #: Optimization status.
    status: OptimizerStatus = OptimizerStatus.SUCCESS
    #: Number of iterations.
    niter: int = 0
    #: Number of cost function calls.
    funcalls: int = 0
    #: Number of gradient function calls.
    gradcalls: int = 0

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return len(self.params)


class Optimizer(Protocol):
    """A protocol class for optimizers."""

    @abstractmethod
    def get_init_state(self, init_params: Params) -> OptimizerState:
        """Returns an initial state for optimization."""
        ...

    @abstractmethod
    def step(
        self,
        state: OptimizerState,
        cost_function: CostFunction,
        grad_function: Optional[GradientFunction] = None,
    ) -> OptimizerState:
        """Run a single optimization step and returns a new state."""
        ...

from typing import TYPE_CHECKING, Callable, cast

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401


# f_tol as create_ftol
def create_ftol(ftol: float) -> Callable[[float, float], bool]:
    """Returns a function evaluating cost function tolerance.
    The return value is True when the cost function difference is less than ``ftol``;
    specifically ``|cost_prev - cost| <= 0.5 * ftol * (|cost_prev| + |cost|) + 1e-20``.
    """

    def fn(cost: float, cost_prev: float) -> bool:
        return (
            2.0 * abs(cost - cost_prev) <= ftol * (abs(cost) + abs(cost_prev)) + 1e-20
        )

    return fn

def gtol(gtol: float) -> Callable[["npt.NDArray[np.float_]"], bool]:
    """Returns a function evaluating gradient function tolerance.
    The return value is True when ``amax(abs(grad)) <= gtol``.
    """

    def fn(grad: "npt.NDArray[np.float_]") -> bool:
        return cast(float, np.amax(np.abs(grad))) <= gtol

    return fn


@dataclass(frozen=True)
class OptimizerStateSPSA(OptimizerState):
    rng: np.random.Generator = np.random.default_rng()
    cost_history: Sequence[float] = field(default_factory=list)

class SPSA(Optimizer):

    def __init__(
        self,
        a: float = 10.0,
        c: float = 0.2,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 0.0,
        ptol: int = 10,
        cost_calc_need: int = 50,
        history: int = 5,
        ftol: Optional[float] = 1e-5,
        rng_seed: Optional[int] = None,
    ):
        if not 0.0 < a:
            raise ValueError(f"'a' must be larger than zero, But got {a}")
        if not 0.0 < alpha:
            raise ValueError(f"'alpha' must be larger than zero. But got {alpha}")
        if not 0.0 < gamma:  # NOTE: is this if statement really neccesary?
            raise ValueError(f"'gamma' must be larger than zero, But got {gamma}")
        
        self._A = A
        
        if a == 10.0:
            self.a = 0.05 * (self._A + 1) ** alpha
        
        self._a = a
        self._c = c
        self._alpha = alpha
        self._gamma = gamma
        self._history = history

        self._best = 100.0
        self._rng_seed = rng_seed

        self._ftol: Optional[Callable[[float, float], bool]] = None
        if ftol is not None:
            if not 0.0 < ftol:
                raise ValueError("ftol must be a positive float.")
            self._ftol = create_ftol(ftol)
        
        self._ptol = ptol # tolerance of plateu steps
        self._p_count = 0
        self._cost_calc_need = cost_calc_need

    def get_init_state(self, init_params: Params) -> OptimizerStateSPSA:
        params = readonly_array(np.array(init_params))
        rng = np.random.default_rng(seed=self._rng_seed)
        cost_history = []
        return OptimizerStateSPSA(
            params=params,
            rng=rng,
            cost_history=cost_history,
        )

    def step(
        self,
        state: OptimizerState,
        cost_function: CostFunction,
        grad_function: Optional[GradientFunction] = None,
    ) -> OptimizerStateSPSA:
        if not isinstance(state, OptimizerStateSPSA):
            raise ValueError('state must have type "OptimizerStateSPSA".')

        rng = state.rng
        cost_history = state.cost_history

        funcalls = state.funcalls
        gradcalls = state.gradcalls
        niter = state.niter + 1

        params = state.params.copy()

        if niter == 1:
        #    cost_prev = cost_function(params)
        #    funcalls += 1
            cost_prev = 0.0
            self._best = 100.0
        else:
            cost_prev = state.cost

        delta = 2 * rng.integers(2, size=state.n_params) - 1
        #delta = np.random.choice([-1,0,1], size=state.n_params, p=[0.45, 0.45, 0.1])

        ck = self._c / (state.niter + 1) ** self._gamma
        params_plus = params + delta * ck
        params_minus = params - delta * ck

        # estimate of the first order gradient
        g = (
            (cost_function(params_plus) - cost_function(params_minus))
            / (2 * ck) * delta
        )
        funcalls += 2

        ak = self._a / (state.niter + self._A + 1) ** self._alpha
        new_params = params - ak * g
        
        # blocking
        if niter > self._cost_calc_need:
            cost = cost_function(params)
            cost_history.append(cost)
            funcalls += 1
            if len(cost_history) > self._history and cost_prev + 2.7*np.std(cost_history[-self._history-1:-1]) < cost:
                new_params = params
                cost = cost_prev
            if cost < self._best:
                self._p_count = 0
                self._best = cost
                self._best_p = readonly_array(params)
                self._best_index = niter
                print(f"best score!! | {niter}: {self._best}")
            else:
                self._p_count += 1
        else:
            cost = 0.0
                
        if self._p_count >= self._ptol:
            status = OptimizerStatus.CONVERGED
        else:
            status = OptimizerStatus.SUCCESS
        
        return OptimizerStateSPSA(
            params=readonly_array(new_params),
            cost=cost,
            status=status,
            niter=niter,
            funcalls=funcalls,
            gradcalls=gradcalls,
            rng=rng,
            cost_history=cost_history
        )


#####################################
############## SSA ##################
#####################################


import math
from numpy import full, zeros, count_nonzero
from collections.abc import Collection

from numpy.random import default_rng

from quri_parts.core.operator import CommutablePauliSet, Operator
from quri_parts.core.sampling import PauliSamplingSetting, PauliSamplingShotsAllocator

def _rounddown_to_unit(n: float, shot_unit: int) -> int:
    return shot_unit * math.floor(n / shot_unit)

def create_frugal_shots_allocator(
    terms_ratio: float = 0.75,
    alloc_per_step: int = 3,
    seed: int = 1,
    shot_unit: int = 1,
) -> PauliSamplingShotsAllocator:
    """allocate shots to some of terms of hamiltonian stochastically
    and the number of selected terms increase gradually.
    total shots are decided by base shots and the number of selected terms."""
    
    rng = default_rng(seed)
    
    niter = 0
    shots_list = zeros(1)

    def allocator(
        operator: Operator,
        pauli_sets: Collection[CommutablePauliSet],
        total_shots: int
    ) -> Collection[PauliSamplingSetting]:
        # create closure to memorize step
        nonlocal niter
        nonlocal shots_list
        
        pauli_sets = tuple(pauli_sets)  # to fix the order of elements
        

        if niter % alloc_per_step == 0: 
            pauli_sets_num = len(pauli_sets)
            # decide the number of terms using in current loop
            selected_terms_num = int(pauli_sets_num * terms_ratio)
            # decide the number of shots from total ratio of the number of selected terms
            current_shots = _rounddown_to_unit(total_shots / pauli_sets_num, shot_unit)
            #print(f"term {selected_terms_num} / {pauli_sets_num}")
            selected_terms_id = rng.choice(pauli_sets_num, selected_terms_num, replace=False)
        
            shots_list = zeros(pauli_sets_num, dtype="int32")
            shots_list[selected_terms_id] = current_shots
        
        # equipartition shots for calculating cost
        elif niter % alloc_per_step == alloc_per_step - 1:
            pauli_sets_num = len(pauli_sets)
            shots_per_term = _rounddown_to_unit(total_shots / pauli_sets_num, shot_unit)
            shots_list = full(pauli_sets_num, shots_per_term, dtype="int32")

        niter += 1

        return frozenset(
            {
                PauliSamplingSetting(
                    pauli_set=pauli_set,
                    n_shots=shots_per_term,
                )
                for (pauli_set, shots_per_term) in zip(pauli_sets, shots_list)
            }
        )

    return allocator


#####################################
############## ZNE ##################
#####################################


from typing import Callable, Iterable, NamedTuple, Optional, Sequence, Tuple

from typing_extensions import TypeAlias

from quri_parts.algo.utils import (
    exp_fitting,
    exp_fitting_with_const,
    exp_fitting_with_const_log,
    polynomial_fitting,
)
from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuit,
    UnboundParametricQuantumCircuit,
    QuantumCircuit,
    inverse_gate,
)
from quri_parts.circuit.gate import ParametricQuantumGate

from quri_parts.core.estimator import (
    ParametricQuantumEstimator,
    Estimatable,
    Estimate,
    QuantumEstimator,
)
from quri_parts.core.operator import Operator, is_hermitian
from quri_parts.core.state import ParametricCircuitQuantumState


class _Estimate(NamedTuple):
    value: float
    error: float = np.nan  # The error of zne has been not implemented yet.


#: Interface representing folding methods
FoldingMethod: TypeAlias = Callable[[UnboundParametricQuantumCircuit, float], list[int]]


#: Interface representing zero noise extrapolation methods
ZeroExtrapolationMethod: TypeAlias = Callable[[Iterable[float], Iterable[float]], float]


#: Interface representing scaling methods of circuit
CircuitScaling: TypeAlias = Callable[
    [UnboundParametricQuantumCircuit, float, FoldingMethod], UnboundParametricQuantumCircuit
]


def _get_residual_n_gates(
    circuit: UnboundParametricQuantumCircuit, scale_factor: float
) -> int:
    """Returns the number of gates that need to be additionally folded after
    the gates of the entire circuit has been folded.
    Args:
        circuit: Circuit to be folded.
        scale_factor: : Factor to scale the circuit. A real number that satisfies >= 1.
    """

    if scale_factor < 1:
        raise ValueError("Scale_factor must be greater than or equal to 1.")
    gate_num = len(circuit.gates)
    int_scale_factor = int((scale_factor - 1) / 2)
    diff = scale_factor - (2 * int_scale_factor + 1)
    return int((diff * gate_num) / 2)


# implementations of FoldingMethod
def create_folding_left() -> FoldingMethod:
    """Returns a :class:`FoldingMethod` that gives a list of indices of gates
    to be folded.
    Folding starts from the left of the circuit.
    """

    def folding_left(
        circuit: UnboundParametricQuantumCircuit, scale_factor: float
    ) -> list[int]:
        add_gate_num = _get_residual_n_gates(circuit, scale_factor)
        return list(range(add_gate_num))

    return folding_left


def create_folding_right() -> FoldingMethod:
    """Returns a :class:`FoldingMethod` that gives a list of indices of gates
    to be folded.
    Folding starts from the right of the circuit.
    """

    def folding_right(
        circuit: UnboundParametricQuantumCircuit, scale_factor: float
    ) -> list[int]:
        add_gate_num = _get_residual_n_gates(circuit, scale_factor)
        n_gates = len(circuit.gates)
        return list(range(n_gates - add_gate_num, n_gates))

    return folding_right


def create_folding_random(seed: Optional[int] = None) -> FoldingMethod:
    """Returns a :class:`FoldingMethod` that gives a list of indices of gates
    to be folded.
    The gates to be folded are chosen at random.
    Args:
        seed: Seed for random number generator.
    """

    def folding_random(
        circuit: UnboundParametricQuantumCircuit, scale_factor: float
    ) -> list[int]:
        add_gate_num = _get_residual_n_gates(circuit, scale_factor)
        list_random = np.random.RandomState(seed)
        add_gate_list = list(
            list_random.choice(range(add_gate_num), size=(add_gate_num), replace=False)
        )
        return add_gate_list

    return folding_random


# implementations of CircuitScaling
def scaling_circuit_folding(
    circuit: UnboundParametricQuantumCircuit,
    circuit_params: Sequence[Sequence[float]],
    scale_factor: float,
    folding_method: FoldingMethod,
) -> Tuple[LinearMappedUnboundParametricQuantumCircuit, Sequence[float]]:
    
    #値も返す。

    scaled_circuit = LinearMappedUnboundParametricQuantumCircuit(circuit.qubit_count)
    scaled_params = []
    num_folding_allgates = int((scale_factor - 1) / 2)
    add_gate_list = folding_method(circuit, scale_factor)
    is_parametric = lambda gate: isinstance(gate, ParametricQuantumGate)
    parametric_gate_counter = 0
    parameter_counter = 0
    
    # folding method
    # U = Ld...L1 where Li is a gate or layer
    # fold(U)=U(U†U)^n(Ld†...Ls†)(Ls...Ld)
    # where n=floor((λ-1)/2), s=floor(λ-1 mod 2)(d/2)
    # λ:scale factor
    for i, gate in enumerate(circuit.gates):
        inv_gate = inverse_gate(gate)
        pf = is_parametric(gate)

        if pf:
            gname = gate.name
            if gname == "ParametricRX":
                pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                scaled_circuit.add_ParametricRX_gate(gate.target_indices[0], pn)
                parameter_counter += 1
                pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                scaled_circuit.add_ParametricRX_gate(gate.target_indices[0], pn)
                parameter_counter += 1
                scaled_params.append(-circuit_params[parametric_gate_counter])
                scaled_params.append(circuit_params[parametric_gate_counter])
            elif gname == "ParametricRY":
                pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                scaled_circuit.add_ParametricRY_gate(gate.target_indices[0], pn)
                parameter_counter += 1
                pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                scaled_circuit.add_ParametricRY_gate(gate.target_indices[0], pn)
                parameter_counter += 1
                scaled_params.append(-circuit_params[parametric_gate_counter])
                scaled_params.append(circuit_params[parametric_gate_counter])
            elif gname == "ParametricRZ":
                pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                scaled_circuit.add_ParametricRZ_gate(gate.target_indices[0], pn)
                parameter_counter += 1
                pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                scaled_circuit.add_ParametricRZ_gate(gate.target_indices[0], pn)
                parameter_counter += 1
                scaled_params.append(-circuit_params[parametric_gate_counter])
                scaled_params.append(circuit_params[parametric_gate_counter])            
        else:
            scaled_circuit.add_gate(gate)

        # λ=3,5,7,... → 1,2,3,... additional gates
        for j in range(num_folding_allgates):
            if pf:
                if gname == "ParametricRX":
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRX_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRX_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    scaled_params.append(-circuit_params[parametric_gate_counter])
                    scaled_params.append(circuit_params[parametric_gate_counter])
                elif gname == "ParametricRY":
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRY_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRY_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    scaled_params.append(-circuit_params[parametric_gate_counter])
                    scaled_params.append(circuit_params[parametric_gate_counter])
                elif gname == "ParametricRZ":
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRZ_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRZ_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    scaled_params.append(-circuit_params[parametric_gate_counter])
                    scaled_params.append(circuit_params[parametric_gate_counter])            
            else:
                scaled_circuit.add_gate(inv_gate)
                scaled_circuit.add_gate(gate)
        # the first s gates are added
        if i in add_gate_list:
            if pf:
                if gname == "ParametricRX":
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRX_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRX_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    scaled_params.append(-circuit_params[parametric_gate_counter])
                    scaled_params.append(circuit_params[parametric_gate_counter])
                elif gname == "ParametricRY":
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRY_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRY_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    scaled_params.append(-circuit_params[parametric_gate_counter])
                    scaled_params.append(circuit_params[parametric_gate_counter])
                elif gname == "ParametricRZ":
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRZ_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    pn = scaled_circuit.add_parameter(f"theta_{parameter_counter}")
                    scaled_circuit.add_ParametricRZ_gate(gate.target_indices[0], pn)
                    parameter_counter += 1
                    scaled_params.append(-circuit_params[parametric_gate_counter])
                    scaled_params.append(circuit_params[parametric_gate_counter])            
            else:
                scaled_circuit.add_gate(gate)
                scaled_circuit.add_gate(gate)
        if pf:
            parametric_gate_counter += 1

    return scaled_circuit, np.array(scaled_params)


# implementations of ZeroExtrapolationMethod
def create_polynomial_extrapolate(order: int) -> ZeroExtrapolationMethod:

    def polynomial_extrapolate(
        scale_factors: Iterable[float], exp_values: Iterable[float]
    ) -> float:
        opt_result = polynomial_fitting(scale_factors, exp_values, order, 0)
        return opt_result.parameters[0]

    return polynomial_extrapolate


def create_exp_extrapolate(order: int) -> ZeroExtrapolationMethod:

    def exp_extrapolate(
        scale_factors: Iterable[float],
        exp_values: Iterable[float],
    ) -> float:
        return exp_fitting(scale_factors, exp_values, order, 0).value

    return exp_extrapolate


def create_exp_extrapolate_with_const(
    order: int, constant: float
) -> ZeroExtrapolationMethod:

    def exp_extrapolate_with_const(
        scale_factors: Iterable[float],
        exp_values: Iterable[float],
    ) -> float:
        opt_result = exp_fitting_with_const(
            scale_factors, exp_values, order, constant, 0
        )
        return opt_result.value

    return exp_extrapolate_with_const


def create_exp_extrapolate_with_const_log(
    order: int, constant: float
) -> ZeroExtrapolationMethod:

    def exp_extrapolate_with_const_log(
        scale_factors: Iterable[float],
        exp_values: Iterable[float],
    ) -> float:
        opt_result = exp_fitting_with_const_log(
            scale_factors, exp_values, order, constant, 0
        )
        return opt_result.value

    return exp_extrapolate_with_const_log


def zne(
    obs: Estimatable,
    circuit: UnboundParametricQuantumCircuit,
    circuit_params: Sequence[Sequence[float]],
    estimator: ParametricQuantumEstimator[ParametricCircuitQuantumState],
    scale_factors: Iterable[float],
    extrapolate_method: ZeroExtrapolationMethod,
    folding_method: FoldingMethod,
    min_cost: float
) -> Tuple[float, Sequence[float]]:

    exp_values: Iterable[float] = []
    circuit_states = []
    scaled_circuit_paramas = []
    for i in scale_factors:
        scaled_circuit, scaled_params = scaling_circuit_folding(circuit, circuit_params, i, folding_method)
        circuit_states.append(
            ParametricCircuitQuantumState(scaled_circuit.qubit_count, scaled_circuit)
        )
        scaled_circuit_paramas.append(
            scaled_params
        )
    
    scale_factors_plus = [1.0] + scale_factors
    for i in range(len(scale_factors_plus)):
        if i == 0:
            exp_values.append(min_cost)
        else:
            exp_values.append(estimator(obs, circuit_states[i-1], [scaled_circuit_paramas[i-1]])[0].value.real)

    print(exp_values)
    return extrapolate_method(scale_factors_plus, exp_values), exp_values


def richardson_extrapolation(
    obs: Estimatable,
    circuit: UnboundParametricQuantumCircuit,
    estimator: ParametricQuantumEstimator[ParametricCircuitQuantumState],
    scale_factors: Iterable[float],
    folding_method: FoldingMethod,
    min_cost: float
) -> float:

    richardson_extrapolate = create_polynomial_extrapolate(
        order=len(list(scale_factors)) - 1
    )
    return zne(
        obs, circuit, estimator, scale_factors, richardson_extrapolate, folding_method, min_cost
    )


def create_zne_estimator(
    estimator: ParametricQuantumEstimator[ParametricCircuitQuantumState],
    scale_factors: Iterable[float],
    extrapolate_method: ZeroExtrapolationMethod,
    folding_method: FoldingMethod,
    min_cost: float
) -> ParametricQuantumEstimator[ParametricCircuitQuantumState]:

    def zne_estimator(
        obs: Estimatable,
        state: ParametricCircuitQuantumState,
        params: Sequence[Sequence[float]],
    ) -> Estimate[float]:
        circuit = state.parametric_circuit
        zne_value = zne(
            obs,
            circuit,
            params,
            estimator,
            scale_factors,
            extrapolate_method,
            folding_method,
            min_cost
        )
        return _Estimate(value=zne_value)

    return zne_estimator


#################################
########### VQE #################
#################################


challenge_sampling = ChallengeSampling(noise=True)


def cost_fn(hamiltonian, parametric_state, param_values, estimator):
    estimate = estimator(hamiltonian, parametric_state, [param_values])
    return estimate[0].value.real

def remove_duplicates(list):
    ret_list = []
    [ret_list.append(elem) for elem in list if elem not in ret_list]
    return ret_list

def cost_std(cost_list, min_cost_index):
    min_cost = cost_list[min_cost_index]
    dedup_cost_list = remove_duplicates(cost_list)
    cost_num = len(dedup_cost_list)
    dedup_min_cost_index = dedup_cost_list.index(min_cost)
    if dedup_min_cost_index == 0:
        return np.std(cost_list[:3])
    elif dedup_min_cost_index - 1 == cost_num:
        return np.std(cost_list[-3:])
    else:
        return np.std(cost_list[dedup_min_cost_index-1:dedup_min_cost_index+1])

def vqe(hamiltonian,
        parametric_state,
        estimator,
        init_params,
        optimizer,
        opt_shots,
        zne_shots,
        measurement_factory,
        hardware_type,
        ):
    opt_state = optimizer.get_init_state(init_params)

    def c_fn(param_values):
        return cost_fn(hamiltonian, parametric_state, param_values, estimator)

    def g_fn(param_values):
        grad = parameter_shift_gradient_estimates(
            hamiltonian, parametric_state, param_values, estimator
        )
        return np.asarray([i.real for i in grad.values])
    costs = []
    min_cost = 0.0
    crr_time = 0.0
    pre_time = 0.0
    time_per_step = 0.0
    scale_factors = [3.0]
    while True:
        try:
            opt_state = optimizer.step(opt_state, c_fn, g_fn)
            print(f"iteration {opt_state.niter}")
            print(opt_state.cost)
            costs.append(opt_state.cost)
            if opt_state.cost < min_cost:
                min_cost = opt_state.cost
                min_cost_params = opt_state.params
        except TimeExceededError:
            print("Reached the limit of shots")
            return opt_state

        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break

        crr_time = challenge_sampling.total_quantum_circuit_time
        time_per_step = max(time_per_step, crr_time - pre_time)
        pre_time = crr_time
        
        if 1000.0 - crr_time < 3*time_per_step/opt_shots*zne_shots*np.sum(scale_factors) + time_per_step + 1.0:
            print(f"time per step: {time_per_step}")
            print(f"total time: {crr_time}")
            print(f"Reached max step({opt_state.niter})")
            break

    dedup_costs = remove_duplicates(costs)
    c_mean = np.mean(dedup_costs)
    c_std = np.std(dedup_costs)
    print(f"mean: {c_mean}")
    print(f"std: {c_std}")
    upper_outlier_cost = c_mean - 3*c_std
    lower_outlier_cost = c_mean - 3.7*c_std
    print(f"upper outlier costs: {upper_outlier_cost}")
    print(f"lower outlier costs: {lower_outlier_cost}")

    # choose an extrapolation method
    extrapolate_method = create_polynomial_extrapolate(order=1)
    # choose how folding your circuit
    folding_method = create_folding_right()
    zne_shots_allocator = create_equipartition_shots_allocator()
    
    zne_sampling_estimator = (
            challenge_sampling.create_concurrent_parametric_sampling_estimator(
                zne_shots, measurement_factory, zne_shots_allocator, hardware_type
            )
    )

    # construct estimator by using zne (only concurrent estimator can be used)
    zne_estimator = create_zne_estimator(
        zne_sampling_estimator, scale_factors, extrapolate_method, folding_method, min_cost
    )
    
    zne_estimated_value1 = zne_estimator(hamiltonian, parametric_state, min_cost_params)
    zne_cost1, zne_costs1 = zne_estimated_value1.value
    zne_cost1 = min(zne_cost1, min(zne_costs1))
    print(f"zne_estimated_value1: {zne_cost1} ")
    
    zne_estimated_value2 = zne_estimator(hamiltonian, parametric_state, min_cost_params)
    zne_cost2, zne_costs2 = zne_estimated_value2.value
    zne_cost2 = min(zne_cost2, min(zne_costs2))
    print(f"zne_estimated_value2: {zne_cost2} ")
    
    zne_estimated_value3 = zne_estimator(hamiltonian, parametric_state, min_cost_params)
    zne_cost3, zne_costs3 = zne_estimated_value3.value
    zne_cost3 = min(zne_cost3, min(zne_costs3))
    print(f"zne_estimated_value3: {zne_cost3}")

    cost_cands = [min_cost]
    
    #return opt_state
    if lower_outlier_cost < zne_cost1 <= upper_outlier_cost:
        cost_cands.append(zne_cost1)
    if lower_outlier_cost < zne_cost2 <= upper_outlier_cost:
        cost_cands.append(zne_cost2)
    if lower_outlier_cost < zne_cost3 <= upper_outlier_cost:
        cost_cands.append(zne_cost3)
    
    if len(cost_cands) == 1:
        dist3 = (zne_cost1 - upper_outlier_cost)**2 + (zne_cost2 - upper_outlier_cost)**2 + (zne_cost3 - upper_outlier_cost)**2 
        dist4 = (zne_cost1 - lower_outlier_cost)**2 + (zne_cost2 - lower_outlier_cost)**2 + (zne_cost3 - lower_outlier_cost)**2
        if dist3 < dist4:
            if (zne_cost1 + zne_cost2 + zne_cost3 - 3*upper_outlier_cost) / 3 > 0.6*c_std:
                return min(min_cost, upper_outlier_cost)
            else:
                return min(min_cost, c_mean - 3.4*c_std)
        else:
            if (-zne_cost1 - zne_cost2 - zne_cost3 + 3*upper_outlier_cost) / 3 > 0.4*c_std:
                return min(min_cost, lower_outlier_cost)
            else:
                return min(min_cost, c_mean - 3.4*c_std)
    else:
        return min(cost_cands)


class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self) -> tuple[Any, float]:
        energy_final = self.get_result()
        qc_time_final = challenge_sampling.total_quantum_circuit_time

        return energy_final, qc_time_final

    def get_result(self) -> Any:
        n_site = 4
        n_qubits = 2 * n_site
        ham = load_operator(
            file_name=f"{n_qubits}_qubits_H_5",
            #data_directory="../hamiltonian",
            data_directory="../hamiltonian/hamiltonian_samples",
            plain_text=False,
        )
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)
        # bk_hamiltonian = bravyi_kitaev(ham)
        # hamiltonian = operator_from_openfermion_op(bk_hamiltonian)

        # make hf + HEreal ansatz
        hf_gates = ComputationalBasisState(n_qubits, bits=0b00001111).circuit.gates
        hf_circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits).combine(hf_gates)
        hw_ansatz = HardwareEfficientReal(qubit_count=n_qubits, reps=1)
        #hw_ansatz = HardwareEfficient(qubit_count=n_qubits, reps=1)
        hf_circuit.extend(hw_ansatz)
        
        parametric_state = ParametricCircuitQuantumState(n_qubits, hf_circuit)
        
        # sc　⇒ must use ZNE
        hardware_type = "it"
        
        # init_terms = 380
        # terms_inc_rate = 0.12
        # shots_allocator = create_stochastic_shots_allocator(init_terms=init_terms, terms_inc_rate=terms_inc_rate, alloc_per_step=2)
        terms_ratio = 0.63 # adjust to calculate gradient stabely
        shots_allocator = create_frugal_shots_allocator(terms_ratio=terms_ratio)
        measurement_factory = bitwise_commuting_pauli_measurement
        n_shots = 6300 # adjust to calculate gradient stably
        zne_shots = 8800

        sampling_estimator = (
            challenge_sampling.create_concurrent_parametric_sampling_estimator(
                n_shots, measurement_factory, shots_allocator, hardware_type
            )
        )
        
        measurements = (measurement_factory(hamiltonian))
        measurements = [m for m in measurements if m.pauli_set != {PAULI_IDENTITY}]
        hm_terms = len(measurements)
        # cost_calc_need = np.ceil((hm_terms - init_terms) * terms_inc_rate)
        # memo alpha:b c:b a:l
        optimizer = SPSA(a=5e-5, c=5e-4, alpha=0.602, gamma=0.201, history=5, A=4.0, ptol=150, cost_calc_need=0)
        #optimizer = Adam(ftol=10e-5)

        init_param = np.random.rand(hw_ansatz.parameter_count) * np.pi * 2e-3

        result = vqe(
            hamiltonian,
            parametric_state,
            sampling_estimator,
            init_param,
            optimizer,
            n_shots * terms_ratio * 2 + n_shots,
            zne_shots,
            measurement_factory,
            hardware_type,
        )
        #print(f"iteration used: {result.niter}")
        #print(f"final params: {result.params}")
        #print(f"best params: {optimizer._best_p}")
        #print(f"best cost: {optimizer._best}")
        #return result.cost
        return result


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())
