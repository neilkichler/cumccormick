#include "cumccormick.cuh"

// #include <iostream>

int main()
{
    cu::mccormick<double> x = 1.0;
    cu::mccormick<double> y = { 1.0, 2.0 };
    auto res = x + y;
    // std::cout << x + y << '\n';
    return 0;
}
