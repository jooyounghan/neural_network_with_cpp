#pragma once
#include <thread>

const int cores = std::thread::hardware_concurrency() / 2;

constexpr int batch_size = 15;