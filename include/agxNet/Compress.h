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


#include <agx/Vector.h>
#include <agx/Referenced.h>
#include <agx/agxCore_export.h>
#include "zlib.h"


namespace agxNet
{

  /** Class for compressing data given a bit-stream. zlib is used for compression.

  Usage: DataRef compressed = Compress::compress(charPtr, numBytes, level);

  compressed will now contain the compressed data. Accessing raw compressed bit-stream is done by:


  char *ptr = compressed->data();
  size_t numBytes = compressed->size();

  ZLIB default compression level is the default.


  Decompression is done by:

  DataRef decompressed = Compress::decompress(compressed->data(), compressed->size());

  (if you don't have a Data object)

  or:

  DataRef decompressed = Compress::decompress(compressed);

  If you have the compressed bit-stream in a Data object.

  */
  class AGXCORE_EXPORT Compress {


  public:

    /// Default constructor
    Compress();

    /// Destructor
    ~Compress();


    /**
    Class storing compressed/decompressed bit-streams as a ref-counted object.
    */
    AGX_DECLARE_POINTER_TYPES(Data);
    AGX_DECLARE_VECTOR_TYPES(Data);
    class Data : public agx::Referenced
    {
    public:
      enum Mode
      {
        NONE,
        COMPRESSED,
        DECOMPRESSED
      };

      /// Specifies a compression level. Default is zlib default compression level.
      enum Level
      {
        DEFAULT_COMPRESSION=-1,
        MIN_COMPRESSION=0,
        MAX_COMPRESSION=9
      };

      /// \return the current mode for the data, compressed/decompressed/none
      Mode getMode() const { return m_mode; }


      /**
      Default constructor
      \param m - the mode of the data
      */
      Data( Mode m=NONE );

      /**
      Constructor which will create a Data object which contains a copy of the source data in compressed form.
      I.e. mode will be COMPRESSED
      */
      Data( const char *source, size_t numBytes );

      /// \return the number of bytes in the data vector
      size_t size() const { return m_data.size(); }

      /// \return a pointer to the bit-stream (compressed or non-compressed).
      char *data() { return m_data.ptr(); }

      /// \return a const pointer to the bit-stream (compressed or non-compressed).
      const char *data() const { return m_data.ptr(); }

      /// \return a pointer to the bit-stream (compressed or non-compressed).
      char *data(size_t idx) { return &m_data[idx]; }

      /// \return a const pointer to the bit-stream (compressed or non-compressed).
      const char *data(size_t idx) const { return &m_data[idx]; }

      /// \return true if the data is valid, compression/uncompression was succesfully done.
      bool valid() const { return m_valid; }

      /// \return the compression ratio of the data, for decompressed data it returns 1.0
      float ratio() const { return m_ratio; }

    protected:

      /// \return index of the current position in the bit-stream
      size_t curr() const { return m_curr; }

      /// set the current position in the bit stream (determines the pointer address of the returned value from currData()
      void setCurr(size_t curr ) { if (curr > m_data.size()) m_curr = std::min( (size_t)0, m_data.size()-1); else m_curr = curr; }

      /// Resize the data-buffer to fit numBytes
      void allocate( size_t numBytes) { if (m_curr >= numBytes) m_curr = numBytes-1; m_data.resize(numBytes); }

      /// \return a pointer to the data-buffer at the current index (set by setCurr).
      char *currData() { return &m_data[m_curr]; }

      /// \return a const pointer to the data-buffer at the current index (set by setCurr).
      const char *currData() const { return &m_data[m_curr]; }


      /// destructor
      virtual ~Data();

      /// Initialize the compression/decompression with a specified compression level (only for compression)
      int init(int level=DEFAULT_COMPRESSION);
      friend class Compress;

      /// Make data valid and call any finalization of the compression/decompression structures.
      void finalize();

      void trim();

    private:


      z_stream *m_strm;
      Mode m_mode;

      agx::Vector<char> m_data;

      size_t m_originalSize;

      bool m_initialized;
      size_t m_curr;
      bool m_valid;
      float m_ratio;
    };



    /**
    Compress a bit-stream with size numBytes.
    \param source - pointer to the bit-stream
    \param numBytes - Number of bytes in the input bit-stream
    \param level - the desired compression level, default is the default zlib compression level
    \return ref_ptr to a Data object with the compressed data. If compression failed nullptr (null) is returned.
    */
    static DataRef compress( const char *source, size_t numBytes, int level=-1);


    /**
    Decompress a bit-stream with size numBytes.
    \param source - pointer to the bit-stream
    \param numBytes - Number of bytes in the input bit-stream
    \return ref_ptr to a Data object with the de-compressed data. If de-compression failed nullptr (null) is returned.
    */
    static DataRef decompress( const char *source, size_t numBytes );

    /**
    Decompress a compressed Data object.
    \param source - pointer to the Data object containing the compressed data.
    \return ref_ptr to a Data object with the de-compressed data. If de-compression failed nullptr (null) is returned.
    */
    static DataRef decompress( Data *source);

  };


} // namespace agxNet

