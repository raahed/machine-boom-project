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

#ifndef AGXCFG_UTILS_H
#define AGXCFG_UTILS_H

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4786) // Disable warnings about long names
#endif

#include <agx/agxCore_export.h>



// Check which compiler we are using
// If it is a gnu compiler of version lower than 3 than use old iostream libs.
#ifdef __GNUC__
#if __GNUC__ <= 2
#define HAVE_NEW_IOSTREAM 0
#else
#define HAVE_NEW_IOSTREAM 1
#endif

#else
#define HAVE_NEW_IOSTREAM 1 // Default
#endif

#include <string>
///

#include <agxCFG/ConfigValue.h>
#include <agx/Vector.h>

namespace agxCFG
{

  ///
  void AGXCORE_EXPORT GetWord( std::string& line, std::string& word );

  ///
  void AGXCORE_EXPORT PeakWord( const std::string& in_line, std::string& word  );

  ///
  void AGXCORE_EXPORT Trim( std::string& str );

  ///
  // Moved to ConfigIO::
  //void ValidValue( const std::string& value, ConfigValue::ValueType &type );
  ///
  // Moved to ConfigIO::
  void AGXCORE_EXPORT ValidKey( const std::string& key );

  ///
  bool AGXCORE_EXPORT StringToFloat( const std::string& word, agx::Real& val );

  /// Utility function for streaming a real value in ascii format. Infinity will be streamed as 1.#INF
  template <typename T>
  std::ostream& streamReal( std::ostream& str, T val )
  {
    T sign = agx::sign( val );
    const char* s_sign = sign == 1 ? "" : "-";
    if (std::isinf(val))
      str << s_sign << "1.#INF";
    else
      str << val;

    return str;
  }

  ///
  void AGXCORE_EXPORT RemoveCntrlChar( std::string& line );
  ///
  int AGXCORE_EXPORT cmp_nocase( const std::string& s1, const std::string& s2 );

  void AGXCORE_EXPORT StripPathDelimiter( std::string& model_path );

  std::string AGXCORE_EXPORT CompoundPath( const std::string& path, const std::string& file );
  std::string AGXCORE_EXPORT AddPathDelimiter( const std::string& path );

  /**
  Using the \p delimiter tokenizer will break up \p str into substrings and put them in \p tokens
  \param str - String to be parsed
  \param delimiter - A string separating the tokens in \p str
  \param tokens - Vector where the result will be put
  */
  void AGXCORE_EXPORT tokenizer( const std::string& str, const std::string& delimiter, agx::Vector<std::string>& tokens );


  /// Return a unique filename that can be used for opening temporary files.
  std::string AGXCORE_EXPORT getTempFileName();

  /**
  Returns a random number in the interval low to high.
  Uses rand(), so seed is dependent on global srand settings.
  */
  double AGXCORE_EXPORT RandInterval( double low = 0, double high = 1 );

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
