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

    void train(const Variable& input, const Variable&label, const float& lr, const int& iters) {
        for (int iter = 0; iter < iters; ++iter) {
            Variable result = funcQueue[0]->forward(input);
            result = result - label;
            funcQueue[0]->backward(result, lr);
        }
    }
    
    Variable forward() {
        // return the result of forward propagation
    }

    ~Model() {
        for (int i = 0; i < funcQueue.size(); ++i) {
            if (funcQueue[i] != nullptr) {
                funcQueue[i]->reset();
                delete funcQueue[i];
                funcQueue[i] = nullptr;
            }
        }
        std::cout << "Model Destructed" << std::endl;
    }
};