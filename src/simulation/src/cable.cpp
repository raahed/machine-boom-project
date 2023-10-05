#include "cable.hpp"
#include <limits>

const std::vector<double> Cable::getLowestNode()
{
    std::size_t lowestNode;
    double lowestZ = std::numeric_limits<double>::infinity();
    int i = 0;
    for (auto itr = this->begin(); itr != this->end(); ++itr, ++i)
    {
        agx::Vec3 pos = itr->getCenterPosition();
        double currentY = pos.z();
        if (currentY < lowestZ) 
        {
            lowestZ = currentY;
            lowestNode = i;
        }
    }
    agx::Vec3 pos = (this->begin() + i)->getCenterPosition();
    std::vector<double> lowestPoint{pos.x(), pos.y(), pos.z()};
    return lowestPoint;
}