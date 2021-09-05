#pragma once
#include "Function.h"
#include "Optimizer.h"
#include <vector>

class Model
{
public:
    std::vector<Function*> funcQueue;
    Optimizer* optimizer_method;

public:
    void addFunc(Function& f)
    {
        if (funcQueue.size() > 0) {
            funcQueue[funcQueue.size() - 1]->next = &f;
            f.back = funcQueue[funcQueue.size() - 1];
        }
        funcQueue.push_back(&f);
    }

    void setOptimizer(Optimizer* op)
    {
        if (optimizer_method != nullptr) {
            delete optimizer_method;
        }
        optimizer_method = op;
    }

    void train(const int& iters, Variable& input, Variable& label) {
        for (int iter = 1; iter <= iters; ++iter) {
            Variable result = funcQueue[0]->forward(input);
            result = result - label;
            if (!(iter % 100)) {
                float error = result.error_sum();
                std::cout << "Iteration " << iter << " : " << error << std::endl;

                if (error < ERROR_MAX) {
                    break;
                }
            }
            funcQueue[funcQueue.size() - 1]->backward(result, lr, optimizer_method);
        }
        std::cout << "train finished" << std::endl;
    }
};