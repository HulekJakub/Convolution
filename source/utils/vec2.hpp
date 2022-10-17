#pragma once
#include <iostream>

namespace utils
{
    template <class T>
    class Vec2
    {
    private:
        T x_;
        T y_;
    public:
        Vec2(T n): x_(n), y_(n){}
        Vec2(T x, T y): x_(x), y_(y){}
        ~Vec2(){}

        T x() const
        {
        return x_;
        }

        T y() const
        {
        return y_;
        }

        void print() const
        {
        std::cout << "(" << x_ << ", " << y_ << ")" << std::endl; 
        }
    };
}
    
