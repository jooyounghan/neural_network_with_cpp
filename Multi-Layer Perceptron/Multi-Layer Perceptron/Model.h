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

    //void train(Variable& input, Variable& label, const int& batch_size, const int& iters, const float& lr)
    void train(Variable& input)
    {
        input.print();
        funcQueue[0]->forward(input);
    }

    ~Model() {
        for (int i = 0; i < funcQueue.size(); ++i) {
            if (funcQueue[i] != nullptr) {
                delete funcQueue[i];
            }
        }
        std::cout << "Model Destructed" << std::endl;
    }
};