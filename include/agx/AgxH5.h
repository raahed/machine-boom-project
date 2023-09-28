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

#ifndef AGX_AGXH5
#define AGX_AGXH5


#include <agx/config/AGX_USE_HDF5.h>
#include <agx/String.h>
#include <agxData/BinaryData.h>
#include <agx/Vector.h>
#include <agx/ThreadSynchronization.h>

#include <iostream>
#include <string.h>
#include <typeinfo>
#include <valarray>

#if AGX_USE_HDF5()
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <H5Cpp.h>
#include <H5Exception.h>

#include <H5Tpublic.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.



#define AGX_HDF5_VERIFY_BUFFER_SIZE 8192
#define agxHdf5VerifyN(x, node, format, ...)                                                                                                                       \
if (!(x))                                                                                                                                                            \
{                                                                                                                                                                    \
  char __str[AGX_HDF5_VERIFY_BUFFER_SIZE];                                                                                                                              \
  agx_snprintf(__str, AGX_HDF5_VERIFY_BUFFER_SIZE, format, ##__VA_ARGS__);                                                                                              \
  ssize_t pathLength = H5Iget_name( (node)->getId(), nullptr, 0 );                                                                                                       \
  agx::String path = "<nameless>";                                                                                                                                      \
  if ( pathLength > 0 )                                                                                                                                      \
  {                                                                                                                                      \
    char* pathBuffer = new char[pathLength+1];                                                                                                                                      \
    ::memset( pathBuffer, 0, pathLength+1 );                                                                                                                                      \
    H5Iget_name( (node)->getId(), pathBuffer, pathLength );                                                                                                                                      \
    path = pathBuffer;                                                                                                                                      \
  }                                                                                                                                      \
  std::stringstream __stream;                                                                                                                                          \
  __stream << "[" << __FILE__ << ":" << __LINE__ << "] HDF5 error: " << __str << ". Error at object " << path << std::endl;                                        \
  __stream << "==============================================================================" << std::endl;                                                           \
  const std::string& __string = __stream.str();                                                                                                                          \
  std::cerr << __string << std::endl;                                                                                                                                  \
  agxThrow std::runtime_error(__string);                                                                                                                                  \
}


namespace agx {
  class Component;

  /**
  The h5 namespace provides functionality for saving/loading data to/from HDF5 files.
  */
  namespace h5 {
    /** TODO:

        Missing in this implementation is a way to handle bitfields, and
        some better way to handle enumerations.

        The "stride" argument used in the journal should be used to set the
        column size of the array being dumped since this is in fact valuable
        information when time comes to read back.

    */



    // HDF5 is not thread safe :/
    extern ReentrantMutex mutex;

    AGXCORE_EXPORT extern const agx::String agx_format;
    AGXCORE_EXPORT extern const agx::String agx_problem_type;
    AGXCORE_EXPORT extern const agx::String agx_problem_mlcp;
    AGXCORE_EXPORT extern const agx::String agx_problem_nlmcp;
    AGXCORE_EXPORT extern const agx::String agx_problem_linear;
    AGXCORE_EXPORT extern const agx::String agx_sparse_matrix;
    AGXCORE_EXPORT extern const agx::String agx_matrix;
    AGXCORE_EXPORT extern const agx::String agx_type;
    AGXCORE_EXPORT extern const agx::String oct_format;
    AGXCORE_EXPORT extern const agx::String oct_type;
    AGXCORE_EXPORT extern const agx::String oct_sparse_matrix;
    AGXCORE_EXPORT extern const agx::String oct_matrix;
    AGXCORE_EXPORT extern const agx::String oct_scalar;
    AGXCORE_EXPORT extern const agx::String oct_value;


    AGXCORE_EXPORT void dumpComponent(agx::Component *component, const agx::String& filePath);


    struct IteratorData {
      IteratorData() : m_name() {}
      agx::String m_name;
    };

    /** A simple wrapper class for the annoying hsize_t dims[2] arrays which
        are used all the time. */
    class AGXCORE_EXPORT h2s  {
    public:
      h2s(hsize_t a, hsize_t b) {
        m[0]=a;
        m[1]=b;
      }
      h2s(hsize_t a) {
        m[0]= a;
        m[1]=0;
      }
      h2s() { m[0]=1; m[1]=1;}
      #if !defined(__APPLE__)
      #ifndef _WIN32
      /// \todo Fix mac & windows compiling
      hsize_t& operator[](size_t i) { return m[i]; }
      hsize_t operator[](size_t i) const  { return m[i]; }
      #endif
      #endif
      operator const hsize_t*() const { return m; }
      operator hsize_t*() { return m; }

      hsize_t m[2];
    };


    // Type information needed by HDF5 functions collected here to allow
    // extensive polymorphism.
    struct h5_type {
      const std::type_info& info;
      const H5::PredType& predicate;
      const H5::DataType type;
      const char* name;
      h5_type()
        : info(typeid(void*)), predicate(H5::PredType::NATIVE_UINT64),
        type(H5::IntType(H5::PredType::NATIVE_UINT64)), name("UNKNOWN")
      {
      }
      h5_type(const std::type_info& newInfo, const H5::PredType& newPredicate,
        const H5::DataType newType, const char* newName)
        : info(newInfo), predicate(newPredicate), type(newType), name(newName)
      {}

      private:
        // Do not use.
        h5_type& operator= (const h5_type&)
        {
          return *this;
        }
    };

    AGXCORE_EXPORT
    const h5_type& h5_type_find(const std::type_info & type);


    AGXCORE_EXPORT
    void dump_string(H5::CommonFG& g, const agx::String& name, const agx::String& s);

    /// Easily create a data space for either scalars or arrays
    AGXCORE_EXPORT H5::DataSpace  create_dataspace(const h2s& h);


    template <typename R>
    void dump_array(H5::CommonFG& g, const std::string& name, const R* a, size_t m=1, size_t n=1)
    {
      try {
        const h5_type& t = h5_type_find(typeid(a[0]));
        if  ( typeid(a[0]) == t.info ) {
          H5::DataSet dataset = g.createDataSet( name, t.type, create_dataspace(h2s(m, n)));
          dataset.write( a, t.predicate);
          dataset.close();
        }
      }catch ( const H5::DataSetIException& exception ) {
        exception.printError();
      }
    }

    template <typename R>
    void overwrite_scalar(H5::CommonFG& g, const agx::String& name, const R a) {
      try {
        const h5_type& t = h5_type_find(typeid(a));
        if ( typeid(a) == t.info ) {
          H5::DataSet dataset = g.openDataSet(name);
          dataset.write(&a, t.predicate);
          dataset.close();
        }
      } catch ( const H5::DataSetIException& exception ) {
        exception.printError();
      }
    }

    template <typename R>
    void dump_scalar(H5::CommonFG& g, const agx::String& name, const R a){
      dump_array<R>(g,  name, &a, 1, 0);
    }

    template <>
    inline void dump_scalar<bool>(H5::CommonFG& g, const agx::String& name, const bool a) {
      int8_t valueBits = static_cast<int8_t>(a);
      try {
        const h5_type& t = h5_type_find(typeid(valueBits));
        if ( typeid(valueBits) == t.info ) {
          H5::DataSet dataset = g.createDataSet( name, t.type, create_dataspace(h2s(1,0)) );
          dataset.write( &valueBits, t.predicate );
          dataset.close();
        }
      } catch ( const H5::DataSetIException& exception ) {
        exception.printError();
      }
    }

    template <typename R>
    void set_scalar_attribute (H5::H5Object& g, const agx::String& name, const R& val){
      try {
        const h5_type& t = h5_type_find(typeid(val));
        if  ( typeid(val) == t.info ) {
          H5::Attribute attr = g.createAttribute(name, t.type, H5::DataSpace(H5S_SCALAR));
          attr.write(t.type, &val);
        } else {
        }
      }
      catch ( const H5::AttributeIException& exception ) {
        exception.printError();
        return;
      }
      catch ( const H5::DataTypeIException& exception ) {
        exception.printError();
        return;
      }
    }

    template<>
    AGXCORE_EXPORT
    void set_scalar_attribute(H5::H5Object& g, const agx::String &name, const agx::String& val);


    AGXCORE_EXPORT
    void print_attributes( H5::H5Object& loc, const H5std_string attr_name, void* operator_data);


    /**
       Read the value of the attribute with the given name from the given group.
       \return True if a write was made to val, false if there is no such attribute,
       or if the attribute was of the wrong type. */
    template < typename T >
    bool get_scalar_attribute(H5::H5Object& g, const agx::String& name, T& result){
      try {
        const h5_type& t = h5_type_find(typeid(result));
        H5::Attribute attribute;
        try{
          attribute = g.openAttribute(name);
        } catch ( const H5::AttributeIException& /*exception*/ ) {
          return false;
        }
        try {
          if ( attribute.getDataType() == t.predicate ) {
            attribute.read( attribute.getDataType(), &result );
            return true;
          }
        } catch ( const H5::DataTypeIException& /*exception*/ ) {
          return false;
        }
      } catch ( const H5::DataTypeIException& /*exception*/ ) {
        return false;
      }
      return false;
    }


    /**
       Read the value of the attribute with the given name from the given group.
       \return True if a write was made to val, false if there is no such attribute,
       or if the attribute was of the wrong type. */
    template <>
    AGXCORE_EXPORT
    bool get_scalar_attribute<agx::String>(H5::H5Object& g, const agx::String& name, agx::String& result);


    /** Read a scalar in a data set with the given name from the given group. */
    template <typename R>
    bool get_scalar(H5::CommonFG& g, const agx::String& name, R& result)
    {
      const h5_type &  t = h5_type_find(typeid(result));
      try {
        H5::DataSet dataset = g.openDataSet(name);
        if  ( dataset.getSpace().getSimpleExtentType() == H5S_SCALAR ) {
          if ( dataset.getDataType() == t.predicate  ) { // need error message or exception if this is false
            try {
              dataset.read(&result, dataset.getDataType() );
              return true;
            } catch ( const H5::DataSetIException& /*exception*/ ) {
              return false;
            }
          } else {
          }
        } else {
          return false;
        }
      } catch ( const H5::DataSetIException& /*exception*/ ) {
        return false;
      }
      return false;
    }


    template<>
    AGXCORE_EXPORT
    inline bool get_scalar<bool>(H5::CommonFG& g, const agx::String& name, bool& result)
    {
      const h5_type& t = h5_type_find(typeid(result));
      try {
        H5::DataSet dataset = g.openDataSet(name);
        if (dataset.getSpace().getSimpleExtentType() == H5S_SCALAR) {
          if (dataset.getDataType() == t.predicate) {
            try {
              int8_t value = 0;
              dataset.read(&value, dataset.getDataType());
              result = (value != 0);
              return true;
            } catch (const H5::DataSetIException& ) {
              return false;
            }
          } else if (dataset.getDataType() == H5::IntType(H5::PredType::STD_I8BE)){
            try {
              dataset.read(&result, dataset.getDataType());
              return true;
            } catch (const H5::DataSetIException& ) {
              return false;
            }
          } else {
            return false;
          }
        } else {
          // Not scalar.
          return false;
        }
      } catch (const H5::DataSetIException&) {
        return false;
      }
    }

    template <typename R>
    /** Dump an array of any type to an open HDF5 group. */
    void dump_array_oct(H5::CommonFG& g, const agx::String& name, const R* a, size_t m=1, size_t n=1)
    {
      H5::Group gg = g.createGroup(name);
      set_scalar_attribute(gg, oct_format, 1);
      set_scalar_attribute(gg, agx_format, 1);
      if ( m>1 || n>1 ) {
        dump_string(gg, oct_type, oct_matrix);
      }
      else {
        dump_string(gg, oct_type, oct_scalar);
      }
      if ( m == n && n == 1)
          n = 0;
      dump_array<R>(gg, "value", a, m, n);
      gg.close();
    }


    /** Dump a scalar of any type to an open HDF5 group. */
    template <typename R>
    void dump_scalar_oct(H5::CommonFG& g, const agx::String& name, R a)
    {
        dump_array_oct<R>(g, name, &a);

    }

    /**
       Read the data set with the given name from the given group. Return nullptr on error.
    */
    AGXCORE_EXPORT agxData::BinaryData* get_data(const H5::CommonFG& group, const agx::String& name);



    AGXCORE_EXPORT H5::H5File* append_or_create(const agx::String& filename);
    AGXCORE_EXPORT H5::Group  append_problem(H5::H5File* file, const agx::String& name = "Problem");

    /**
    \return The index of the child with the given name. Returns parent.getNumObjs() if no such child exists.
    */
    AGXCORE_EXPORT hsize_t getIndexOf( const H5::CommonFG& parent, const H5std_string& childName );
    AGXCORE_EXPORT bool hasChildNamed( const H5::H5Object& parent, const H5std_string& childName );
    AGXCORE_EXPORT bool hasChildNamed( const H5::H5File& parent, const H5std_string& childName );


    /**
       Return the child group with the given name. Will be created if it doesn't exist.
    */
    AGXCORE_EXPORT H5::Group getOrCreateGroup( H5::Group& parent, const H5std_string& childName );
    AGXCORE_EXPORT H5::Group getOrCreateGroup( H5::H5File* parent, const H5std_string& childName );

    /**  A utility that just checks for the existence of an attribute with given name */
    AGXCORE_EXPORT bool check_attr(const H5::H5Object& g, const agx::String & name);

    /** Read the string dataset of given name into the agx::String buff.
        Return 0 if not found, 1 for success.  */
    AGXCORE_EXPORT int get_string(const H5::CommonFG& g, const agx::String& name, agx::String& buff );


    /**
        Read an array and put content in object v which must be a class with a method
        resize(size_t n) and operator *  which returns the data buffer.
     */
    template<typename R>
    int get_array(const H5::CommonFG& g, const agx::String& name, R& v )
    {
      int ret = 0;
      try {
        H5::Exception::dontPrint();
        H5::DataSet t  = g.openDataSet(name);
        try {
          H5::Exception::dontPrint();
          H5::DataSpace ds = t.getSpace();
          h2s dims;
          ds.getSimpleExtentDims(dims);
          v.resize(size_t(dims[0]));
          t.read(&(v[0]), t.getFloatType());
          t.close();
          ret = 1;
        } catch ( H5::DataSetIException& /*error*/ ) {
        }
      }
      catch (H5::GroupIException& /*error*/ ) {
      }
      return ret;
    }


    template <typename R>
    int  get_array_oct(const H5::CommonFG& g, const agx::String& name, R& v  )
    {
      return get_array<R>(g, name, v);
    }

  } /* namespace H5 */
} /* namespace agx */

extern "C" {
  herr_t list_nodes (hid_t group_id, const char* member_name, void* operator_data);
}


// AGX_USE_HDF5.
#endif

// Include guard.
#endif
