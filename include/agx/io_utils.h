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

#ifndef AGX_IO_UTILS_H
#define AGX_IO_UTILS_H

#include <agx/Vec2.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>
#include <agx/Quat.h>


namespace agx
{

  //////////////////////////////////////////////////////////////////////////
  // Vec2 streaming operators
  inline std::ostream& operator << ( std::ostream& output, const Vec2& vec )
  {
    output << vec[0] << " " << vec[1];
    return output;     // to enable cascading
  }

  inline std::istream& operator >> ( std::istream& input, Vec2& vec )
  {
    input >> vec[0] >> vec[1];
    return input;
  }


  //////////////////////////////////////////////////////////////////////////
  // Vec3i steaming operators.
  inline std::ostream& operator << ( std::ostream& output, const Vec3i& vec )
  {
    output << vec[0] << " "
    << vec[1] << " "
    << vec[2];
    return output;     // to enable cascading
  }

  inline std::istream& operator >> ( std::istream& input, Vec3i& vec )
  {
    input >> vec[0] >> vec[1] >> vec[2];
    return input;
  }


  //////////////////////////////////////////////////////////////////////////
  // Vec3d steaming operators.
  inline std::ostream& operator << ( std::ostream& output, const Vec3d& vec )
  {
    output << vec[0] << " "
      << vec[1] << " "
      << vec[2];
    return output;     // to enable cascading
  }

  inline std::istream& operator >> ( std::istream& input, Vec3d& vec )
  {
    input >> vec[0] >> vec[1] >> vec[2];
    return input;
  }


  //////////////////////////////////////////////////////////////////////////
  // Vec3f steaming operators.
  inline std::ostream& operator << ( std::ostream& output, const Vec3f& vec )
  {
    output << vec[0] << " "
      << vec[1] << " "
      << vec[2];
    return output;     // to enable cascading
  }

  inline std::istream& operator >> ( std::istream& input, Vec3f& vec )
  {
    input >> vec[0] >> vec[1] >> vec[2];
    return input;
  }



  //////////////////////////////////////////////////////////////////////////
  // Vec4 steaming operators.
  inline std::ostream& operator << ( std::ostream& output, const Vec4& vec )
  {
    output  << vec[0] << " "
    << vec[1] << " "
    << vec[2] << " "
    << vec[3];
    return output;     // to enable cascading
  }

  inline std::istream& operator >> ( std::istream& input, Vec4& vec )
  {
    input >> vec[0] >> vec[1] >> vec[2] >> vec[3];
    return input;
  }





  //////////////////////////////////////////////////////////////////////////
  // Matrix steaming operators.
#ifndef SWIG_CPPWRAPPER_BUILD
  inline std::ostream& operator<< (std::ostream& os, const Matrix& m)
  {
    os << "{" << std::endl;
    for ( int row = 0; row < 4; ++row ) {
      os << "\t";
      for ( int col = 0; col < 4; ++col )
        os << m( row, col ) << " ";
      os << std::endl;
    }
    os << "}" << std::endl;
    return os;
  }
#endif
  //////////////////////////////////////////////////////////////////////////
  // Quat streaming operators.
  inline std::ostream& operator << ( std::ostream& output, const Quat& vec )
  {
    output << vec[0] << " "
    << vec[1] << " "
    << vec[2] << " "
    << vec[3];
    return output;     // to enable cascading
  }

  inline std::istream& operator >> ( std::istream& input, Quat& vec )
  {
    input >> vec[0] >> vec[1] >> vec[2] >> vec[3];
    return input;
  }



}    // end of namespace agx
#endif
