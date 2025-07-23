#include <cumccormick.cuh>

int main()
{
    cu::mccormick<double> x = 1.0;
    cu::mccormick<double> y = { 1.0, 2.0 };
    auto res = x + y;
    return 0;
}
