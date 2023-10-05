#ifndef CABLE_HPP
#define CABLE_HPP

#include <vector>
#include <agxCable/Cable.h>

struct Cable : public agxCable::Cable {
    const std::vector<double> getLowestNode();
};

#endif