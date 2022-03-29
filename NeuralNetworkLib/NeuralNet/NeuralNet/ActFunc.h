#pragma once
#include "Neuron.h"

class ActFunc
{
public:
	virtual void Do(std::vector<Neuron>& in, std::vector<Neuron>& out) = 0;
};

class Sigmoid : public ActFunc
{
public:
	void Do(std::vector<Neuron>& in, std::vector<Neuron>& out) override
	{
		const int32 inCnt = in.size();
		ASSERT_CRASH(inCnt == out.size())
		for (int idx = 0; idx < inCnt; ++idx)
		{
			double& outVal = out[idx].value;
			double& inVal = in[idx].value;

			outVal = 1.0 / (1.0 + std::exp(inVal));
		}
		return;
	}
};

class Relu : public ActFunc
{
public:
	void Do(std::vector<Neuron>& in, std::vector<Neuron>& out) override
	{
		const int32 inCnt = in.size();
		ASSERT_CRASH(inCnt == out.size())
			for (int idx = 0; idx < inCnt; ++idx)
			{
				double& outVal = out[idx].value;
				double& inVal = in[idx].value;

				if (inVal > 0)
					outVal = inVal;
				else
					outVal = 0;
			}
		return;
	}

};