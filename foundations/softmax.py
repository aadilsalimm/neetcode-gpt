import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_el = max(z)
        exp_sum = 0
        for n in z:
            exp_sum += np.exp(n - max_el)
        
        res = []
        for n in z:
            res.append(round(((np.exp(n - max_el)) / exp_sum), 4))
        
        return np.array(res, dtype=np.float64)