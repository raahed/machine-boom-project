#ifndef CONSTRAINTS_HPP
#define CONSTRAINTS_HPP

// check https://www.stroustrup.com/bs_faq2.html#constraints
// This struct checks whether the template parameter T can be converted to B
template<class T, class B>
struct Convertible_to {
    static void constraints(T *p) { B *pb = p; }

    Convertible_to() { void (*p)(T *) = constraints; }
};

#endif