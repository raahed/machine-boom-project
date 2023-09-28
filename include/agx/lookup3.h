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

/*
This source code has been taken and modified by Algoryx Simulation AB
from the source and under the license given below.
*/

/*
-------------------------------------------------------------------------------
lookup3.c, by Bob Jenkins, May 2006, Public Domain.

These are functions for producing 32-bit hashes for hash table lookup.
hashword(), hashlittle(), hashlittle2(), hashbig(), mix(), and final()
are externally useful functions.  Routines to test the hash are included
if SELF_TEST is defined.  You can use this free for any purpose.  It's in
the public domain.  It has no warranty.

You probably want to use hashlittle().  hashlittle() and hashbig()
hash byte arrays.  hashlittle() is is faster than hashbig() on
little-endian machines.  Intel and AMD are little-endian machines.
On second thought, you probably want hashlittle2(), which is identical to
hashlittle() except it returns two 32-bit hashes for the price of one.
You could implement hashbig2() if you wanted but I haven't bothered here.

If you want to find a hash of, say, exactly 7 integers, do
  a = i1;  b = i2;  c = i3;
  mix(a,b,c);
  a += i4; b += i5; c += i6;
  mix(a,b,c);
  a += i7;
  final(a,b,c);
then use c as the hash value.  If you have a variable length array of
4-byte integers to hash, use hashword().  If you have a byte array (like
a character string), use hashlittle().  If you have several byte arrays, or
a mix of things, see the comments above hashlittle().

Why is this so big?  I read 12 bytes at a time into 3 4-byte integers,
then mix those integers.  This is fast (you can do a lot more thorough
mixing with 12*3 instructions on 3 integers than you can with 3 instructions
on 1 byte), but shoehorning those bytes into integers efficiently is messy.
-------------------------------------------------------------------------------
*/

#ifndef AGX_LOOKUP3_H
#define AGX_LOOKUP3_H


#include <agx/agx.h>
#include <agx/Math.h>
#include <agx/agxPhysics_export.h>

namespace agx {

  /**
  --------------------------------------------------------------------
   This works on all machines.  To be useful, it requires
   -- that the key be an array of uint32_t's, and
   -- that the length be the number of uint32_t's in the key

   The function hashword() is identical to hashlittle() on little-endian
   machines, and identical to hashbig() on big-endian machines,
   except that the length has to be measured in uint32_ts rather than in
   bytes.  hashlittle() is more complicated than hashword() only because
   hashlittle() has to dance around fitting the key bytes into registers.
  --------------------------------------------------------------------
  */
  AGXPHYSICS_EXPORT uint32_t hashword(
  const uint32_t *k,              /* the key, an array of uint32_t values */
  size_t          length,         /* the length of the key, in uint32_ts */
  uint32_t        initval) ;      /* the previous hash, or an arbitrary value */



  /**
  --------------------------------------------------------------------
  hashword2() -- same as hashword(), but take two seeds and return two
  32-bit values.  pc and pb must both be non-null, and *pc and *pb must
  both be initialized with seeds.  If you pass in (*pb)==0, the output
  (*pc) will be the same as the return value from hashword().
  --------------------------------------------------------------------
  */
  AGXPHYSICS_EXPORT void hashword2(
  const uint32_t *k,              /* the key, an array of uint32_t values */
  size_t          length,         /* the length of the key, in uint32_ts */
  uint32_t       *pc,             /* IN: seed OUT: primary hash value */
  uint32_t       *pb);          /* IN: more seed OUT: secondary hash value */

}


#endif
