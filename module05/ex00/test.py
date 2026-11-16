from module05.ex00.martix import Matrix, Vector

a = Matrix([[1, 2], [3, 4]])
b = Matrix([[5, 6], [7, 8]])
assert (a+b).data == [[6.,8.],[10.,12.]]
assert (a-b).data == [[-4.,-4.],[-4.,-4.]]
assert (a*b).data == [[19.,22.],[43.,50.]]
assert (a*2).data == [[2.,4.],[6.,8.]]
assert a.T().data == [[1.,3.],[2.,4.]]
assert isinstance(a*Vector([[2],[3]]),Vector)
assert Vector([[1],[2]]).dot(Vector([[3],[4]])) == 11.
assert isinstance(Vector([[1],[2]]) + Vector([[3],[4]]),Vector)
print('ex00 tests passed')
