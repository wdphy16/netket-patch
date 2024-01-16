from dataclasses import dataclass
from typing import Any, Optional, Protocol

from netket.optimizer.linear_operator import LinearOperator, SolverT
from netket.utils.types import PyTree
from netket.vqs import VariationalState


class LHSConstructorAdaptT(Protocol):
    def __call__(self, vstate: VariationalState, diag_shift: float) -> LinearOperator:
        pass


@dataclass
class LinearPreconditionerAdapt:
    lhs_constructor: LHSConstructorAdaptT
    solver: SolverT
    solver_restart: bool = False
    x0: Optional[PyTree] = None
    info: Any = None

    def __call__(self, vstate, gradient, step):
        lhs = self.lhs_constructor(vstate=vstate, diag_shift=vstate.diag_shift)
        x0 = self.x0 if self.solver_restart else None
        self.x0, self.info = lhs.solve(self.solver, gradient, x0=x0)
        return self.x0
