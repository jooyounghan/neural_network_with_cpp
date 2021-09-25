#pragma once
#include <thread>

constexpr int seed = 100;

const int cores = std::thread::hardware_concurrency();