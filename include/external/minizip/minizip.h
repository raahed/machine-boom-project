/* zlib.h -- interface of the 'zlib' general purpose compression library
  version 1.2.8, April 28th, 2013

  Copyright (C) 1995-2013 Jean-loup Gailly and Mark Adler

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  Jean-loup Gailly        Mark Adler
  jloup@gzip.org          madler@alumni.caltech.edu


  The data format used by the zlib library is described by RFCs (Request for
  Comments) 1950 to 1952 in the files http://tools.ietf.org/html/rfc1950
  (zlib format), rfc1951 (deflate format) and rfc1952 (gzip format).
*/

#ifndef MINIZIP_H
#define MINIZIP_H

#ifdef __cplusplus
extern "C" {
#endif

  /*
  Usage : minizip [-o] [-a] [-0 to -9] [-p password] [-j] file.zip [files_to_add]

  -o  Overwrite existing file.zip
  -a  Append to existing file.zip
  -0  Store only
  -1  Compress faster
  -9  Compress better

  -j  exclude path. store only the file name.
  */

int agx_minizip(int argc, const char *argv[]);

#ifdef __cplusplus
}
#endif


#endif /* MINIZIP_H */

