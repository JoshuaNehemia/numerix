from typing import Callable
from numerix.Numerix import Numerix


class Secant(Numerix):
    """
    Secant method for finding roots of a single-variable function.
    """

    def __init__(
        self,
        function: Callable,
        x0: float,
        x1: float,
        *,
        is_verbose: bool = False,
        is_logging: bool = False,
    ):
        super().__init__(is_verbose=is_verbose, is_logging=is_logging)
        self.set_function(function)
        self.set_initial_points(x0, x1)

    def set_function(self, function: Callable):
        if not callable(function):
            raise TypeError("Function must be callable.")

        self.function = function
        self.args_count = function.__code__.co_argcount

        if self.args_count != 1:
            raise ValueError("Secant method supports only f(x).")

    def set_initial_points(self, x0: float, x1: float):
        if x0 == x1:
            raise ValueError("Initial points must be different.")

        self.x0 = float(x0)
        self.x1 = float(x1)

    def start(self, tol: float = 1e-6, max_iter: int = 1000) -> float:
        f = self.function
        x_prev, x_curr = self.x0, self.x1
        f_prev, f_curr = f(x_prev), f(x_curr)

        for i in range(1, max_iter + 1):
            if f_curr == f_prev:
                raise ZeroDivisionError("Division by zero in secant update.")

            x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            f_next = f(x_next)

            if self.is_logging:
                self.add_logs(
                    {
                        "iter": i,
                        "x_prev": x_prev,
                        "x_curr": x_curr,
                        "x_next": x_next,
                        "f(x)": f_next,
                    }
                )

            if abs(x_next - x_curr) < tol or abs(f_next) < tol:
                return x_next

            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_next, f_next

        raise RuntimeError("Secant method did not converge.")
