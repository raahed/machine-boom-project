/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

/*
This source code has been taken and modified by Algoryx Simulation AB
from the source and under the license given below.
*/

/*
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2007 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com


This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#ifndef AGXCOLLIDE_PRIMITIVE_INDEX_PAIR_H
#define AGXCOLLIDE_PRIMITIVE_INDEX_PAIR_H


/// \cond INTERNAL_DOCUMENTATION

namespace agxCollide
{

  /// A class for storing indices to a pair of primitives, e.g. triangles.
  /// The indices are supposed to point to a storage lying outside the class each.
  class PrimitiveIndexPair
  {
  public:

    PrimitiveIndexPair()
    {}

    /** Constructor
    * \param p - another triangle pair
    */
    PrimitiveIndexPair( const PrimitiveIndexPair& p )
    {
      m_index1 = p.m_index1;
      m_index2 = p.m_index2;
    }

    /** Constructor
    * \param index1 - index of first primitive
    * \param index2 - index of second primitive
    */
    PrimitiveIndexPair( unsigned int index1, unsigned int index2 )
    {
      m_index1 = index1;
      m_index2 = index2;
    }

    /**
    * Get index of first primitive
    */
    AGX_FORCE_INLINE unsigned int getIndex1() const
    {
      return m_index1;
    }

    /**
    * Get index of second primitive
    */
    AGX_FORCE_INLINE unsigned int getIndex2() const
    {
      return m_index2;
    }

  private:
    unsigned int m_index1;
    unsigned int m_index2;
  };

  /// For storing a vector of primitive pairs
  typedef agx::Vector<PrimitiveIndexPair> PrimitiveIndexPairVector;

}

#endif

/// \endcond
