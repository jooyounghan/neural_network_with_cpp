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
        if (funcQueue.size() > 0) {
            funcQueue[funcQueue.size() - 1]->next = f;
            f->back = funcQueue[funcQueue.size() - 1];
        }
        funcQueue.push_back(f);
    }

    void train(const float& lr, const int& iters, Variable& input, Variable& label) {
        for (int iter = 1; iter <= iters; ++iter) {
            Variable result = funcQueue[0]->forward(input);
            result = result - label;
            if (!(iter % 100)) {
                float error = result.sum();
                std::cout << "Iteration " << iter << " : " << error << std::endl;

                if (error < ERROR_MAX) {
                    break;
                }
            }
            funcQueue[funcQueue.size() - 1]->backward(result, lr);
        }
        std::cout << "train finished" << std::endl;
    }


    ~Model() {
        for (int i = 0; i < funcQueue.size(); ++i) {
            if (funcQueue[i] != nullptr) {
                funcQueue[i]->reset();
                delete funcQueue[i];
                funcQueue[i] = nullptr;
            }
        }
    }
};