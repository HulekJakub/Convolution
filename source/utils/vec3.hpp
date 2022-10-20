#pragma once
#include <iostream>

namespace utils
{
    template <typename T> struct vec3_type { typedef T type; };

    template <class T>
    class Vec3
    {
    private:
        T x_;
        T y_;
        T z_;

        int mul(vec3_type<int>) const
        { 
            return x_ * z_ * y_;
        }

    public:
        Vec3(){}
        Vec3(T n): x_(n), y_(n), z_(n){}
        Vec3(T x, T y, T z): x_(x), y_(y), z_(z){}
        ~Vec3(){}

        T x() const
        {
            return x_;
        }

        T y() const
        {
            return y_;
        }

        T z() const
        {
            return z_;
        }
                
        T mul() const
        { 
            return mul(vec3_type<T>()); 
        }

        void print() const
        {
            std::cout << "(" << x_ << ", " << y_ << ", " << z_ << ")" << std::endl; 
        }

        inline bool operator==(const Vec3<T>& rhs) const
        { 
            return x_ == rhs.x_ && y_ == rhs.y_ && z_ == rhs.z_;
        }
        inline bool operator!=(const Vec3<T>& rhs) const
        { 
            return !(*this == rhs); 
        }

    };
       
}
