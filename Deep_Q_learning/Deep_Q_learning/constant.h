#pragma once
#include <thread>

const int cores = std::thread::hardware_concurrency();