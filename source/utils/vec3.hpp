#pragma once
#include <iostream>

namespace utils
{
    template <class T>
    class Vec3
    {
    private:
        T x_;
        T y_;
        T z_;
    public:
        Vec3(T n): x_(n), y_(n), z_(n){}
        Vec3(T x, T y, T z): x_(x), y_(y), z_(z){}
        ~Vec3(){}

        T x()
        {
            return x_;
        }

        T y()
        {
            return y_;
        }

        T z()
        {
            return z_;
        }

        void print() const
        {
            std::cout << "(" << x_ << ", " << y_ << ", " << z_ << ")" << std::endl; 
        }
    };
       
}
