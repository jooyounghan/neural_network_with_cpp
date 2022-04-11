#include <Matrix.h>

int main()
{
	CMatrix a = CMatrix(1024, 1024);
	a.XavierNormalInitialize(3, 4);

	CTimer timer;
	
	timer.StartCheck();
	double* b = a.TransposeSerial();
	timer.EndCheck();
	timer.PrintTime("Serial");

	timer.StartCheck();
	double* c = a.TransposeParallel();
	timer.EndCheck();
	timer.PrintTime("Parallel");
}