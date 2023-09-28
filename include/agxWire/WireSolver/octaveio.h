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

#include<iostream>
#include<fstream>
#include<sstream>
#include<iomanip>

#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace octaveio{

  template <class T>
  std::ostream & write_vector( std::ostream & s, std::string name, const T& x){
    s << "# name: "       << name     << std::endl;
    s << "# type: matrix"             << std::endl;
    s << "# rows: "       << x.size() << std::endl;
    s << "# columns: 1"               << std::endl;

    for ( size_t i = 0; i < x.size(); ++i ){
      s <<  ( double ) x[ i ] << std::endl;
    }

    return s;
  }

  template <class T>
  std::ostream & write_scalar( std::ostream & s, std::string name, const T& x){

    s << "# name: "       << name << std::endl;
    s << "# type: scalar"         << std::endl;
    s << ( double ) x             << std::endl;

    return s;

  }


  inline std::ostream & write_struct( std::ostream & s, std::string name, size_t n){

    s << "# name: "              << name << std::endl;
    s << "# type: scalar struct"         << std::endl;
    s << "# ndims: 2"                    << std::endl;
    s << " 1 1"                          << std::endl;
    s << "# length: "            << n    << std::endl;

    return s;

  }

  template <class T>
  std::ostream & write_vector_patched( std::ostream & s, const T& x, size_t N, bool pre = false){
    size_t n = N - x.size();

    s << "  ";

    if ( pre ){
      for ( size_t i = 0; i < n; ++i )
        s << 0.0 << "  "  ;
    }

    for ( size_t i = 0; i < x.size(); ++i ){
      s << x[ i ] << "  ";
    }

    if ( ! pre ){
      for ( size_t i = 0; i < n; ++i )
        s << 0.0 << "  "  ;
    }
    s << std::endl;

    return s;

  }



  template <class T>
  std::ostream & write_vector3( std::ostream & s, std::string name, const T& x){
    s << "# name: "       << name     << std::endl;
    s << "# type: matrix"             << std::endl;
    s << "# rows: "       << x.size() << std::endl;
    s << "# columns: 3"               << std::endl;

    for ( size_t i = 0; i < x.size(); ++i ){
      s <<  ( double ) x[ i ][ 0 ] << " " << ( double ) x[ i ][ 1 ] << " " << ( double ) x[ i ][ 2 ] << std::endl;
    }

    return s;
  }

  template<class T, class S>
  void write_octave_wire_config( const std::string & filename,
                                         const T& lengths,
                                         const T& tensions,
                                         const S& edge_normals,
                                         const S& segment_normals,
                                         std::string prefix = std::string("wire_config_") ) {
    static size_t i = 0;
    std::stringstream name;
    name << prefix << std::setw(4) << std::setfill('0') << i++;
    std::fstream fs;
    if ( i == 0 )
      fs.open (filename, std::fstream::in|std::fstream::out|std::fstream::trunc);
    else
      fs.open (filename, std::fstream::in|std::fstream::out|std::fstream::app);

    octaveio::write_struct(fs, name.str(), 4 ); // 4 members in the struct
    octaveio::write_vector(fs, "lengths", lengths);
    octaveio::write_vector(fs, "tensions", tensions);
    octaveio::write_vector3(fs, "edge_normals", edge_normals);
    octaveio::write_vector3(fs, "segment_normals", segment_normals);
    fs.close();
  }




} // namespace octaveio

DOXYGEN_END_INTERNAL_BLOCK()
