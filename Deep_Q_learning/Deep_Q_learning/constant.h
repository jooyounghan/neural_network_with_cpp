#pragma once
#include <thread>

const int cores = std::thread::hardware_concurrency() / 2;

constexpr int input_size = 4;