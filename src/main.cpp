#include "matrix.h"


int main(int argc, char** argv) {
	Matrix<int16_t, 2, 3> mat(-2);
	Matrix<int16_t, 3, 3> mat2(4);

	mat[0] = 9;

	mat.print();
	mat2.print();

	mat = mat * mat2;
	mat.print();
	mat.abs();
	mat.print();

	// uint64_t n = 4294967296 * 1;
	// printf("%lld\n", n);

	return 0;
}