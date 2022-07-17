#include <pch.h>
#include <CMatrix.h>
#include <CInitializer.h>

int main()
{
	CMatrix a{ 2, 1 };
	CMatrix b{ 1, 3 };
	CInitializer::NormalInitialize(a);
	CInitializer::NormalInitialize(b);
	a.PrintData();
	b.PrintData();
	CMatrix c = a.MatMul(b);
	c.PrintData();
	c.Transpose().PrintData();
}