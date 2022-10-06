from dataclasses import dataclass
from typing import Any, Callable, Optional

from netket.optimizer.linear_operator import LinearOperator, SolverT
from netket.utils.types import PyTree
from netket.vqs import VariationalState

LHSConstructorAdaptT = Callable[[VariationalState, float], LinearOperator]


@dataclass
class LinearPreconditionerAdapt:
    lhs_constructor: LHSConstructorAdaptT
    solver: SolverT
    solver_restart: bool = False
    x0: Optional[PyTree] = None
    info: Any = None
    _lhs: LinearOperator = None

    def __init__(self, lhs_constructor, solver, solver_restart=False):
        self.lhs_constructor = lhs_constructor
        self.solver = solver
        self.solver_restart = solver_restart

    def __call__(self, vstate, gradient, diag_shift):
        self._lhs = self.lhs_constructor(vstate=vstate, diag_shift=diag_shift)
        x0 = self.x0 if self.solver_restart else None
        self.x0, self.info = self._lhs.solve(self.solver, gradient, x0=x0)
        return self.x0

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"\n\tlhs_constructor = {self.lhs_constructor}, "
            + f"\n\tsolver          = {self.solver}, "
            + f"\n\tsolver_restart  = {self.solver_restart},"
            + ")"
        )
