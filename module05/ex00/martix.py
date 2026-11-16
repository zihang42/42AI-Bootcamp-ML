class Matrix:
    def __init__(self, data) -> None:
        if isinstance(data, tuple) and all(
            isinstance(d, int) and d > 0 for d in data
        ):
            data = [[0.0] * data[0] for _ in range(data[1])]
        if not isinstance(data, list) or not all(
            isinstance(row, list) for row in data
        ):
            raise ValueError("Not valid data")

        self.data: list[list[float]] = [
            [float(x) for x in row] for row in data
        ]
        self.shape: tuple[int, int] = (len(self.data), len(self.data[0]))

    def __add__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError("Shape doesn't match")
        new = [
            [a + b for a, b in zip(r, s)]
            for r, s in zip(self.data, other.data)
        ]
        return type(self)(new)

    def __radd__(self, other: Matrix) -> Matrix:
        return other.__add__(self)

    def __sub__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError("Shape doesn't match")
        new = [
            [a - b for a, b in zip(r, s)]
            for r, s in zip(self.data, other.data)
        ]
        return Matrix(new)

    def __rsub__(self, other: Matrix) -> Matrix:

        return other.__sub__(self)

    def __truediv__(self, other: int | float) -> Matrix:
        if other == 0:
            raise ZeroDivisionError("Division by zero")
        new = [[val / other for val in row] for row in self.data]
        return Matrix(new)

    def __rtruediv__(self, other: int | float) -> Matrix:
        raise TypeError("Not defined")

    def __mul__(self, other: Matrix | int | float) -> Matrix:
        if isinstance(other, (int, float)):
            new = [[val * other for val in row] for row in self.data]
            return Matrix(new)
        if not isinstance(other, Matrix):
            raise TypeError("Expected scalar or Matrix")
        if self.shape[1] != other.shape[0]:
            raise ValueError("Incompatible matrix shapes")
        result = [
            [
                sum(
                    self.data[i][k] * other.data[k][j]
                    for k in range(self.shape[1])
                )
                for j in range(other.shape[1])
            ]
            for i in range(self.shape[0])
        ]
        if isinstance(other, Vector):
            return Vector(result)

        return Matrix(result)

    def __rmul__(self, other: Matrix | int | float) -> Matrix:
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        elif isinstance(other, Matrix):
            return other.__mul__(self)
        else:
            raise NotImplementedError

    def T(self: Matrix) -> Matrix:
        return Matrix([list(row) for row in (zip(*self.data))])

    def __str__(self):
        return str(self.data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.data})"


class Vector(Matrix):
    def __init__(self, data) -> None:
        super().__init__(data)
        if self.shape[0] != 1 and self.shape[1] != 1:
            raise ValueError("Vector must be a single row or column")

    def dot(self, v: Vector) -> float:
        if self.shape != v.shape:
            raise ValueError("Vectors must have the same shape")
        return float(
            sum(
                a * b
                for ra, rb in zip(self.data, v.data)
                for a, b in zip(ra, rb)
            )
        )
