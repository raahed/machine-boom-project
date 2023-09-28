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

#ifndef SABRE_KERNELS_H
#define SABRE_KERNELS_H

#include <agxSabre/agxSabre.h>


namespace agxSabre
{


  /**
  Helper class to choose which computational kernels to use.

  Hardware is probed and the instruction sets that can be used can be seen by getHardwareFeatures. If AVX2 is
  supported, it is a very good idea to leave it enabled.

  There should not be a need for overriding the kernel choice, but if wanted it is possible by
  creating a value by or:ing together constants from IsMask and then call setPreferredVectorInstructionSet.

  E.g:
    SabreKernels::setPreferredVectorInstructionSet( SabreKernels::SCALAR_BIT | SabreKernels::SSE3_BIT );
    Which would disallow fused-multiply-add even if the hardware supports it.



  */
  class AGXSABRE_EXPORT SabreKernels
  {
    public:

      /**
      Bitmasks for hardware instruction sets used for SabreKernels get/set methods.
      */
      enum IsMask {
        SCALAR_BIT  = 0x01, /**< Fallback */
        SSE3_BIT    = 0x02, /**< SSE */
        AVX1_BIT    = 0x04, /**< AVX1 */
        AVX2_BIT    = 0x08, /**< AVX2, adds FMA instructions */
        AVX512F_BIT = 0x10  /**< AVX-512 */
      };



      /**
      Constructor. There shouldn't be a user need to create any instance of this class.
      */
      SabreKernels();



      /**
      Returns a number with bits set according to IsMask indicating detected hardware features.
      */
      inline static unsigned int getHardwareFeatures() { return s_hwFeatures; }


      /**
      Returns a value used by Sabre to select appropriate kernels.
      */
      inline static unsigned int getPreferredVectorInstructionSet() { return s_preferredVecIS; }


      /**
      Override the preferred vector instruction set value for which kernels to use.

      The input value will be masked with the available hardware features to prevent
      enabling instruction sets which the CPU does not support which would cause
      program abortion due to illegal instruction.

      \return True if the input value could be used as is. Returns false if some parts of the input value were masked away.
      */
      static bool setPreferredVectorInstructionSet( unsigned int instructionSets );


    private:
      static unsigned int s_preferredVecIS;
      static unsigned int s_hwFeatures;
  };




}

#endif

