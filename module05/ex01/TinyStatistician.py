import numpy as np


class TinyStatistician:
    def _check(self, x: np.ndarray | list) -> list[float] | None:
        if not isinstance(x, (np.ndarray, list)):
            return
        try:
            res = np.asarray(x)
            if res.size == 0 or res.ndim != 1:
                return
            return [float(v) for v in res]
        except ValueError, TypeError, OverflowError:
            return

    def mean(self, x: np.ndarray | list) -> float | None:
        res = self._check(x)
        if res is None:
            return
        sum = 0.0
        for ele in res:
            sum += ele
        return sum / len(res)

    def percentile(self, x: np.ndarray | list, p: int) -> float | None:
        res = self._check(x)
        if res is None or not 0 < p < 100:
            return
        res.sort()
        pos = (len(res) - 1) * p / 100.0
        lower = int(pos)
        upper = min(lower + 1, len(res) - 1)
        return res[lower] + (res[upper] - res[lower]) * (pos - lower)

    def median(self, x: np.ndarray | list) -> float | None:
        return self.percentile(x, 50)

    def quartile(self, x: np.ndarray | list) -> None | list[float | None]:
        res = self._check(x)
        if res is None:
            return
        return [self.percentile(res, 25), self.percentile(res, 75)]

    def var(self, x: np.ndarray | list) -> float | None:

        mean_val = self.mean(x)
        if mean_val is None:
            return

        sum = 0.0
        for v in x:
            sum += (v - mean_val) ** 2
        return sum / len(x)

    def std(self, x: np.ndarray | list) -> float | None:
        if (var_val := self.var(x)) is None:
            return
        return var_val ** (1 / 2)


# def main():
#     t = TinyStatistician()
#     x = np.array([1, 2, 3, 10, 5, 8])

#     print(t.std(x))

# main()
