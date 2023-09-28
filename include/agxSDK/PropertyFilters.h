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

#ifndef AGXSDK_PROPERTYFILTERS_H
#define AGXSDK_PROPERTYFILTERS_H

#include <agx/agxPhysics_export.h>
#include <agx/stdint.h>
#include <agx/Referenced.h>
#include <agxCollide/GeometryPair.h>
#include <agxCollide/Contacts.h>
#include <agxSDK/ExecuteFilter.h>


namespace agx
{
  class PropertyContainer;
}

namespace agxSDK
{

  /**
  A class for filtering contacts based on the geometries' properties
  (as in: their property container).

  Use this class to specify callbacks for certain combinations of geometry collision events.
  */
  class AGXPHYSICS_EXPORT PropertyFilter : public ExecuteFilter
  {
    public:

      /**
      This filter will select contacts that contain a string property of value p data
      */
      PropertyFilter( );

      /**
      Called when two PropertyContainers should be compared.
      \return true when c1 can be considered equal to c2.
      */
      virtual bool matches(const agx::PropertyContainer* c1, const agx::PropertyContainer* c2) const = 0;

      /// This is the function called to filter contacts.
      virtual bool match(const agxCollide::Geometry* geo0, const agxCollide::Geometry* geo1) const override;
      using ExecuteFilter::match;

    protected:

      /// Destructor
      virtual ~PropertyFilter();

      std::string m_property_id[2];

      bool m_is_set[2];
      bool m_has_data[2];
  };

  typedef agx::ref_ptr<PropertyFilter> PropertyFilterRef;


#define MATCH_IMPLEMENTATION( TYPENAME, TYPE ) \
  int checkMatch( const agx::PropertyContainer* c, size_t numMatch) const \
  { \
    if (!c) \
      return 0; \
    size_t i=0; \
    TYPE data; \
    int match1=0; \
    \
    do { \
      if ( m_is_set[i] && c && c->getProperty  ## TYPENAME ( m_property_id[i], data )) \
      { \
        match1 = (m_property_data[i] == data) ? ( i ? 1 : -1) : 0; \
      } \
      i++; \
    } while( !match1 && i < numMatch); \
    return match1; \
  } \
  \
  bool matches(const agx::PropertyContainer* c1, const agx::PropertyContainer* c2) const \
  { \
    size_t n1 = c1 ? c1->getNumberOfProperties  ## TYPENAME () : 0; \
    size_t n2 = c2 ? c2->getNumberOfProperties ## TYPENAME () : 0; \
    \
    size_t numRequiredMatches = (m_is_set[0] ? 1 : 0) + (m_is_set[1] ? 1 : 0); \
    \
    /* No properties what so ever, bail out*/ \
    if (!n1 && !n2) \
      return false; \
    \
    int match1 = 0, match2 = 0; \
    match1 = checkMatch( c1, numRequiredMatches ); \
    match2 = checkMatch( c2, numRequiredMatches ); \
    \
    bool result = (numRequiredMatches == 1 && (match1 || match2) ) || \
                  (numRequiredMatches  == 2 && (match1 && match2) && (match1 != match2) ); \
    \
    return result; \
  }

  /// Class for matching Long int properties
  class AGXPHYSICS_EXPORT LongPropertyFilter : public PropertyFilter
  {
    public:
      LongPropertyFilter( const std::string& property_id, const int64_t& data );

      LongPropertyFilter( const std::string& property_id0, const int64_t& data0,
                          const std::string& property_id1, const int64_t& data1);

      MATCH_IMPLEMENTATION( Long, int64_t );

    protected:

      /// Destructor
      virtual ~LongPropertyFilter();

      int64_t  m_property_data[2];
  };


  /// Class for matching  int properties
  class AGXPHYSICS_EXPORT IntPropertyFilter : public PropertyFilter
  {
    public:
      IntPropertyFilter( const std::string& property_id, const int32_t& data );

      IntPropertyFilter( const std::string& property_id0, const int32_t& data0,
                         const std::string& property_id1, const int32_t& data1);

      MATCH_IMPLEMENTATION( Int, int32_t );

    protected:

      /// Destructor
      virtual ~IntPropertyFilter();

      int32_t  m_property_data[2];
  };

  /// Class for matching  float properties
  class AGXPHYSICS_EXPORT FloatPropertyFilter : public PropertyFilter
  {
    public:
      FloatPropertyFilter( const std::string& property_id, const float& data );

      FloatPropertyFilter( const std::string& property_id0, const float& data0,
                           const std::string& property_id1, const float& data1);

      MATCH_IMPLEMENTATION( Float, float );

    protected:

      /// Destructor
      virtual ~FloatPropertyFilter();


      float  m_property_data[2];
  };


  /// Class for matching  double properties
  class AGXPHYSICS_EXPORT DoublePropertyFilter : public PropertyFilter
  {
    public:
      DoublePropertyFilter( const std::string& property_id, const double& data );

      DoublePropertyFilter( const std::string& property_id0, const double& data0,
                            const std::string& property_id1, const double& data1);

      MATCH_IMPLEMENTATION( Double, double );

    protected:

      /// Destructor
      virtual ~DoublePropertyFilter();

      double  m_property_data[2];
  };


  /// Class for matching  bool properties
  class AGXPHYSICS_EXPORT BoolPropertyFilter : public PropertyFilter
  {
    public:
      BoolPropertyFilter( const std::string& property_id, const bool& data );

      BoolPropertyFilter( const std::string& property_id0, const bool& data0,
                          const std::string& property_id1, const bool& data1);

      MATCH_IMPLEMENTATION( Bool, bool );


    protected:

      /// Destructor
      virtual ~BoolPropertyFilter();

      bool  m_property_data[2];
  };

  /// Class for matching  bool properties
  class AGXPHYSICS_EXPORT StringPropertyFilter : public PropertyFilter
  {
    public:
      StringPropertyFilter( const std::string& property_id, const std::string& data );

      StringPropertyFilter( const std::string& property_id0, const std::string& data0,
                            const std::string& property_id1, const std::string& data1);

      MATCH_IMPLEMENTATION( String, std::string );

    protected:

      /// Destructor
      virtual ~StringPropertyFilter();

      std::string  m_property_data[2];
  };


  inline bool PropertyFilter::match(const agxCollide::Geometry* geo0, const agxCollide::Geometry* geo1) const
  {
    const agx::PropertyContainer* c0 = geo0->hasPropertyContainer() ? geo0->getPropertyContainer() : nullptr;
    const agx::PropertyContainer* c1 = geo1->hasPropertyContainer() ? geo1->getPropertyContainer() : nullptr;
    return matches(c0, c1);
  }

} // namespace agxSDK

#endif


