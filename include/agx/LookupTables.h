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

#ifndef AGX_LOOKUP_TABLES_H
#define AGX_LOOKUP_TABLES_H

#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agx/List.h>
#include <agxStream/Serializable.h>
#include <agxStream/StorageStream.h>

namespace agx
{
  typedef agx::List< std::pair< agx::Real, agx::Real > > RealPairList;
  typedef agx::Vector< std::pair< agx::Real, agx::Real > > RealPairVector;


  /**
  Return a value of y given any value of x

  The variable look up function can be set to calculate any wanted x value.

  The look up table return tabulated (and interpolated) values that
  Default result scaler function is not scaling at all.
  */
  class AGXPHYSICS_EXPORT LookupTable : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Create a lookup table
      */
      LookupTable();

      /**
      clear all data
      */
      virtual void clear();

      /**
      insert a value to the lookup table.
      \param x - the lookup variable to be calculated using lookup
      \param y - the defined value for that specific lookup variable.
      */
      virtual void insertValue( const agx::Real x, const agx::Real y );

      /**
      return the number of values in the lookup table
      */
      size_t getNumValues() const;


      /**
      Lookup a specific value
      */
      virtual agx::Real lookup( const agx::Real x ) const = 0;

      /**
      Set the default return value, if the table has been cleared.
      */
      void setDefaultValue( agx::Real defaultValue );

      /**
      \returns the default return value, if the table has been cleared.
      */
      agx::Real getDefaultValue() const;

      /**
      Set all lookup values at the same time (all previous will be removed.)
      */
      virtual void setValues( RealPairVector const& values );

      /**
      Scale the found value.
      Default vale to scale the looked up value with is 1.
      */
      virtual agx::Real resultScalerFunction() const;


      RealPairList::const_iterator getValueIterator() const;
      RealPairList::const_iterator getValueEndIterator() const;

    protected:
      virtual ~LookupTable();
      agx::Real m_defaultValue;
      RealPairList m_xyPair;
  };


  /**
  Linear interpolation between the inserted variable/value pairs.
  Values looked up outside the set range will be interpolated using the tangent at that end of the table.
  */
  class AGXPHYSICS_EXPORT LinearLookupTable : virtual public LookupTable // virtual inheritance due to diamond
  {
  public:
    /**
    Constructor.
    */
    LinearLookupTable();

    /**
    Lookup a specific value. The result will be interpolated linearly between the values in the table.
    */
    virtual agx::Real lookup( const agx::Real x ) const override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agx::LinearLookupTable);
  protected:
    virtual ~LinearLookupTable();
  };
  typedef agx::ref_ptr<LinearLookupTable> LinearLookupTableRef;

  class Point2D : public agx::Vec3
  {
  public:
    Point2D() {}
    Point2D(agx::Real xxx, agx::Real yyy, agx::Real dy)
    {
      x() = xxx;
      y() = yyy;
      z() = dy;
    }
    Point2D(agx::Vec3 point)
    {
      x() = point.x();
      y() = point.y();
      z() = point.z();
    }
    agx::Real yp() { return z(); }
  };
  typedef agx::Vector<Point2D> Point2DVector;

  class AGXPHYSICS_EXPORT CubicFunction2D : public agx::Referenced, public agxStream::Serializable
  {
  public:
    CubicFunction2D() {}

    virtual void add(agx::Real x, agx::Real y, agx::Real dy);

    virtual agx::Real evaluate(agx::Real x);

    virtual agx::Real evaluateDerivative(agx::Real x);

    virtual void clear() { m_points.clear(); }

    virtual const Point2DVector& getPoints() { return m_points; }

    AGXSTREAM_DECLARE_SERIALIZABLE(agx::CubicFunction2D);

  private:
    Point2DVector::iterator getIteratorAfter(const agx::Real x);
    Point2DVector m_points;
  };
  typedef agx::ref_ptr<CubicFunction2D> CubicFunction2DRef;

  /**
  Base class for spline interpolation between the inserted variable/value pairs
  */
  class AGXPHYSICS_EXPORT CubicSplineLookupTable : virtual public LookupTable
  {
  public:
    CubicSplineLookupTable();

    /**
    Lookup a specific value
    */
    virtual agx::Real lookup(const agx::Real x) const override;

    virtual agx::Real lookupDerivative(const agx::Real x) const;

    virtual void insertValue(const agx::Real x, const agx::Real y) override;

    virtual void clear() override;


    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str);

    AGXSTREAM_DECLARE_SERIALIZABLE(agx::CubicSplineLookupTable);
  protected:
    virtual ~CubicSplineLookupTable() {}
    CubicFunction2DRef m_spline;
  };
  typedef agx::ref_ptr<CubicSplineLookupTable> CubicSplineLookupTableRef;

}

#endif // AGX_LOOKUP_TABLES_H
