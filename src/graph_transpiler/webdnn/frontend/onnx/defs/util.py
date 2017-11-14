from typing import Optional

from webdnn.graph.variable import Variable


def check_broadcast_constraints(a: Variable, b: Variable, axis: Optional[int] = None):
    if axis is None:
        axis = a.ndim - b.ndim

    i_a = axis
    i_b = 0
    while i_a < a.ndim and i_b < b.ndim:
        if a.shape[i_a] == b.shape[i_b] or a.shape[i_a] == 1 or b.shape[i_b] == 1:
            a.order.axes[i_a].unify(b.order.axes[i_b])

            if (a.shape[i_a] == 1 and b.shape[i_b] == 1) or (a.shape[i_a] != 1 and b.shape[i_b] != 1):
                # If broadcast is not occurred, size must be same
                assert a.shape[i_a] == b.shape[i_b], f"""
Shape mismatch: a.shape[{i_a}] != b.shape[{i_b}]
  (a.shape) = {a.shape}
  (b.shape) = {b.shape}
"""

            i_a += 1
            i_b += 1

        else:
            raise ValueError(f"""Broadcast is failed: \n
  (a.shape)={a.shape}
  (b.shape)={b.shape}
  (axis)={axis}""")
