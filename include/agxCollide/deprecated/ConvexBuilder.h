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
#pragma once

#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agx/Vector.h>
#include <agxCollide/Convex.h>
#include <agxIO/MeshReader.h>
#include <agx/agx_vector_types.h>
#include <agx/debug.h>

namespace agxCollide
{
  namespace deprecated {

    typedef agx::Vector<agxCollide::ConvexRef> ConvexRefVector;

    class ConvexBuilderImplementation;
    AGX_DECLARE_POINTER_TYPES(ConvexBuilder);
    /**
    Class for creating convex meshes from a set of vertices and indices.
    First the method build() is called. If this was successful (return > 0)
    the created ConvexShapes will be available in a vector (getConvexShapes())

    Parameters such as: setMaxVertices(), setDepth() etc should be set BEFORE calling build()

   \deprecated Use agxCollide::ConvexBuilder instead
    */
    class AGXPHYSICS_EXPORT ConvexBuilder : public agx::Referenced
    {
    public:

      /// Constructor
      ConvexBuilder();

      /**
      Constructs a number of Convex shapes from a triangle list.
      */
      size_t build( const agx::Vec3Vector& vertices, const agx::UInt32Vector& indices, size_t numTriangles );

      /**
      Constructs a number of Convex shapes from the triangles read with the MeshReader.
      The reader MUST be successfully initialized with data before this call.
      \return number of created Convex meshes, 0 if error occurs.
      */
      size_t build(agxIO::MeshReader* reader);

      /// \return a vector of constructed Convex shapes
      const ConvexRefVector& getConvexShapes() const { return m_convexShapes; }

      /// \return the ith shape
      agxCollide::ConvexRef getShape(size_t i)
      {
        agxAssert(i < m_convexShapes.size());
        if (i >= m_convexShapes.size())
          return nullptr;
        return m_convexShapes[i];
      }

      /// return num shapes
      size_t getNumShapes() const { return m_convexShapes.size(); }

      /// Option: Set maximum number of vertices in the output hull. Recommended 32 or less.
      void setMaxVertices(unsigned int n = 32) { m_maxVertices = n; }

      /// \return max vertices
      unsigned int getMaxVertices() const { return m_maxVertices; }

      /// Option: Set the depth to split, a maximum of 10, generally not over 7.
      void setDepth(unsigned int depth = 5) { m_depth = depth; }
      unsigned int getDepth() const { return m_depth; }

      /// Option: Set the concavity threshold percentage.  0=20 is reasonable.
      void setConcavityPercent(unsigned int concPercent = 5) { m_concavityPercent = concPercent; }
      unsigned int getConcavityPercent() const { return m_concavityPercent; }

      /// Option: Set the percentage volume conservation threshold to collapse hulls. 0-30 is reasonable.
      void setVolumePercent(unsigned int volumePercent = 5) { m_volumePercent = volumePercent; }
      unsigned int getVolumePercent() const { return m_volumePercent; }

      /// Option: Set the skin width to apply to the output hulls.
      void setSkinWidth(double swidth = 0.0) { m_skinWidth = swidth; }
      double getSkinWidth() const { return m_skinWidth; }

    protected:
      virtual ~ConvexBuilder();

      size_t buildConvexShapes(agxCollide::deprecated::ConvexBuilderImplementation* impl);

    private:

      agxIO::MeshReaderRef m_reader;

      double m_skinWidth;
      unsigned int m_volumePercent;
      unsigned int m_concavityPercent;
      unsigned int m_maxVertices;
      unsigned int m_depth;

      ConvexRefVector m_convexShapes;

      ConvexBuilderImplementation* m_impl;
    };
  }
}
