#pragma once
#include "Function.h"
#include <vector>
class Model
{
public:
    std::vector<Function*> funcQueue;

public:
    void addFunc(Function* f)
    {
        funcQueue.push_back(f);
    }

    void train(Variable* input, Variable* label, const int& batch_size, const int& iters, const float& lr)
    {

    }
};