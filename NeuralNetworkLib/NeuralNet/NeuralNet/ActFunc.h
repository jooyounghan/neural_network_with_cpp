#pragma once

template<typename T>
class ActFunc
{
	virtual T* GetResult(const int& numOfElement, T* input) = 0;
	virtual T* GetDeriviate(const int& numOfElement, T* input) = 0;
};

template<typename T>
class Sigmoid : public ActFunc<T>
{
	T* GetResult(T* input)
	{
		
	}
};

template<typename T>
class Relu : public ActFunc<T>
{
	T* GetResult(T* input)
	{

	}
};

template<typename T>
class Identity : public ActFunc<T>
{
	T* GetResult(T* input)
	{

	}
};

template<typename T>
class Softmax : public ActFunc<T>
{
	T* GetResult(T* input)
	{

	}
};