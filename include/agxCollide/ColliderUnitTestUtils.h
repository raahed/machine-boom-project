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

#ifndef AGXUNIT_COLLIDERUNITTESTUTILS_H
#define AGXUNIT_COLLIDERUNITTESTUTILS_H

#include <agx/agxPhysics_export.h>
#include <agx/agx_vector_types.h>
#include <agxUnit/UnitMacros.h>
#include <agx/Referenced.h>
#include <agxCollide/Contacts.h>

DOXYGEN_START_INTERNAL_BLOCK()

namespace agx
{
  class RigidBody;
}

namespace agxSDK {
  class Simulation;
}

namespace agxCollide {

  AGX_DECLARE_POINTER_TYPES(ContactTester);
  class AGXPHYSICS_EXPORT ContactTester : public agx::Referenced
  {
    public:
      virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder)=0;

  };


  AGX_DECLARE_POINTER_TYPES(AnyContactTester);
  class AGXPHYSICS_EXPORT AnyContactTester : public ContactTester
  {
    public:
      AnyContactTester(size_t numAppearances = 1) :
        m_numAppearances(numAppearances)
      {
      }

      virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder) override;

    protected:
      size_t m_numAppearances;
  };

  AGX_DECLARE_POINTER_TYPES(NormalAndDepthContainedTester);
  class AGXPHYSICS_EXPORT NormalAndDepthContainedTester : public ContactTester
  {
    public:
      NormalAndDepthContainedTester(const agx::Vec3& normal, agx::Real depth,
        size_t numAppearances = 1) :
      m_normal(normal), m_depth(depth),
        m_numAppearances(numAppearances)
      {
        m_normal.normalize();
      }

      virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder) override;

    protected:
      agx::Vec3 m_normal;
      agx::Real m_depth;
      size_t m_numAppearances;
  };


  AGX_DECLARE_POINTER_TYPES(NormalTester);
  class AGXPHYSICS_EXPORT NormalTester : public ContactTester
  {
  public:
    NormalTester(const agx::Vec3& normal, size_t numAppearances = 1) :
    m_normal(normal), m_numAppearances(numAppearances)
    {
        m_normal.normalize();
    }

    virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder) override;

  protected:
    agx::Vec3 m_normal;
    size_t m_numAppearances;
  };


  AGX_DECLARE_POINTER_TYPES(NormalDotProductTester);
  /// Tests that found normal * given normal >= given dot product.
  class AGXPHYSICS_EXPORT NormalDotProductTester : public ContactTester
  {
  public:
    NormalDotProductTester(const agx::Vec3& normal, agx::Real dotProduct, size_t numAppearances = 1) :
        m_normal(normal), m_dotProduct(dotProduct), m_numAppearances(numAppearances)
        {
          m_normal.normalize();
        }

        virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder) override;

  protected:
    agx::Vec3 m_normal;
    agx::Real m_dotProduct;
    size_t m_numAppearances;
  };


  AGX_DECLARE_POINTER_TYPES(AllNormalDotProductTester);
  class AGXPHYSICS_EXPORT AllNormalDotProductTester : public ContactTester
  {
  public:
    AllNormalDotProductTester(const agx::Vec3& normal, agx::Real dotProduct) :
      m_normal(normal), m_dotProduct(dotProduct)
    {
      m_normal.normalize();
    }

    virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder) override;

  protected:
    agx::Vec3 m_normal;
    agx::Real m_dotProduct;
  };


  AGX_DECLARE_POINTER_TYPES(AllSameNormalsTester);
  class AGXPHYSICS_EXPORT AllSameNormalsTester : public ContactTester
  {
  public:
    AllSameNormalsTester(const agx::Vec3& normal) :
        m_normal(normal)
        {
          m_normal.normalize();
        }

        virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder) override;

  protected:
    agx::Vec3 m_normal;
  };


  AGX_DECLARE_POINTER_TYPES(NoContactTester);
  class AGXPHYSICS_EXPORT NoContactTester : public ContactTester
  {
  public:
    NoContactTester(){}

    virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder) override;
  };


  AGX_DECLARE_POINTER_TYPES(AllTester);
  class AGXPHYSICS_EXPORT AllTester : public ContactTester {
  public:
    AllTester(const agx::Vec3& point, const agx::Vec3& normal, agx::Real depth) :
    m_point(point), m_normal(normal), m_depth(depth)
    {
      m_normal.normalize();
    }

    virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder) override;

  protected:
    agx::Vec3 m_point;
    agx::Vec3 m_normal;
    agx::Real m_depth;
  };


  AGX_DECLARE_POINTER_TYPES(NormalOrOppositeAndDepthContainedTester);
  class AGXPHYSICS_EXPORT NormalOrOppositeAndDepthContainedTester : public ContactTester {
  public:
    NormalOrOppositeAndDepthContainedTester(const agx::Vec3& normal, agx::Real depth,
      size_t numAppearances = 1) :
    m_normal(normal), m_depth(depth),
      m_numAppearances(numAppearances)
    {
      m_normal.normalize();
    }

    virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder) override;

  protected:
    agx::Vec3 m_normal;
    agx::Real m_depth;
    size_t m_numAppearances;
  };


  AGX_DECLARE_POINTER_TYPES(OrthoNormalAndDepthContainedTester);
  class AGXPHYSICS_EXPORT OrthoNormalAndDepthContainedTester : public ContactTester {
  public:
    OrthoNormalAndDepthContainedTester(const agx::Vec3& normal, agx::Real depth,
      size_t numAppearances = 1) :
    m_normal(normal), m_depth(depth),
      m_numAppearances(numAppearances)
    {
      m_normal.normalize();
    }

    virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder) override;

  protected:
    agx::Vec3 m_normal;
    agx::Real m_depth;
    size_t m_numAppearances;
  };




  AGX_DECLARE_POINTER_TYPES(MeshTriangleTester);
  class AGXPHYSICS_EXPORT MeshTriangleTester : public ContactTester {
  public:
    enum ContactsToTest {TEST_FOR_ONE, TEST_FOR_ALL};

    MeshTriangleTester( ContactsToTest toTest) :
      m_toTest(toTest)
    {}

    /**
    \param triangleNr The number of the triangle in the trimesh
    \param voronoiRegion The voronoi region on the triangle
    \param shapeNr The number of the shape in the contact (0 for from first geo or 1 for second).
    */
    void addContact(agx::UInt32 triangleNr, agx::UInt8 voronoiRegion, agx::UInt8 shapeNr)
    {
      m_triangleNrs.push_back(triangleNr);
      m_voronoiRegions.push_back(voronoiRegion);
      agxAssert( shapeNr < 2);
      m_shapeNumbers.push_back(shapeNr);
    }

    virtual void testContacts(agxCollide::LocalGeometryContact* result, bool inversedOrder) override;

  protected:
    agx::UInt32Vector m_triangleNrs;
    agx::UInt8Vector m_voronoiRegions;
    agx::UInt8Vector m_shapeNumbers;
    ContactsToTest m_toTest;
  };


  AGXPHYSICS_EXPORT void testResting(
    agxSDK::Simulation* simulation,
    const agx::RigidBodyPtrVector& bodies,
    const char* groupName,
    const char* testName,
    agx::Real preTestingTime = agx::Real(1.0),
    bool onlyForDoublePrecision = false);

  inline void testResting(
    agxSDK::Simulation* simulation,
    agx::RigidBody* body,
    const char* groupName,
    const char* testName,
    agx::Real preTestingTime = agx::Real(1.0),
    bool onlyForDoublePrecision = false)
  {
    agx::RigidBodyPtrVector bodies;
    bodies.push_back( body );
    testResting( simulation, bodies, groupName, testName, preTestingTime, onlyForDoublePrecision );
  }


  AGXPHYSICS_EXPORT void testContacts(
    agxSDK::Simulation* simulation,
    const char* groupName,
    const char* testName,
    ContactTester* tester,
    bool onlyForDoublePrecision = false);

  /**
  This function tests that a collider does not violate the bounds in the
  result vector, by modifying contacts which it has not put there itself.
  It does this only for

  \param geo0 The first geometry to test.
  \param geo1 The second geometry to test.

  */
  AGXPHYSICS_EXPORT void testColliderRespectsOtherResults(
    agxCollide::GeometryRef geo0, agxCollide::GeometryRef geo1);
}

DOXYGEN_END_INTERNAL_BLOCK()

#endif
