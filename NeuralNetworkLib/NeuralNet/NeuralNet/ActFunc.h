#pragma once

template<typename T>
class CActFunc
{
	virtual T* GetResult(const int& numOfElement, T* input) = 0;
	virtual T* GetDeriviate(const int& numOfElement, T* input) = 0;
};

template<typename T>
class Sigmoid : public CActFunc<T>
{
	T* GetResult(T* input)
	{
		
	}
};

template<typename T>
class Relu : public CActFunc<T>
{
	T* GetResult(T* input)
	{

	}
};

template<typename T>
class Identity : public CActFunc<T>
{
	T* GetResult(T* input)
	{

	}
};

template<typename T>
class Softmax : public CActFunc<T>
{
	T* GetResult(T* input)
	{

	}
};