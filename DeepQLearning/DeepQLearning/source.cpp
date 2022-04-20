#include <pch.h>
#include "pch.h"
int main()
{
	double* t = new double[6]{ -2, -1, 0, 1, 2, 3 };
	double* tt = new double[6]{ -2, -1, 0, 1, 2, 3 };
	double* ttt = new double[6]{ -2, -1, 0, 1, 2, 3 };
	CMatrix a = CMatrix(2, 3, t);
	CMatrix b = CMatrix(2, 3, tt);
	CMatrix c = CMatrix(2, 3, ttt);

	Sigmoid sig;
	Relu relu;
	Identity identity;

	CTimer timer;

	timer.StartCheck();
	CMatrix* d = sig.GetDeriviate(&a);
	CMatrix* e = relu.GetDeriviate(&b);
	CMatrix* f = identity.GetDeriviate(&c);
	timer.EndCheck();
	timer.PrintElapsedTime("Get Result");

	std::cout << std::endl;
	d->PrintData();
	std::cout << std::endl;
	e->PrintData();
	std::cout << std::endl;
	f->PrintData();
	std::cout << std::endl;
}