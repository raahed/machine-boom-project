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

#ifndef AGX_VECTOR_TYPES_H
#define AGX_VECTOR_TYPES_H

#include <agx/agx.h>
#include <agx/Vector.h>
#include <agx/Integer.h>

#include <agx/Referenced.h>
#include <agx/Name.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>
#include <agx/Quat.h>
#include <agx/AffineMatrix4x4.h>

namespace agx {

  // Types using vector
  class RigidBody;
  class Constraint;
  class SPDMatrix3x3;
  class ElementaryConstraint;
  class Attachment;
  class ConstraintImplementation;
  class FluidContactPoint;

  typedef Vector<Name>                 NameVector;

  typedef VectorPOD<Vec4>              Vec4Vector;
  typedef VectorPOD<Vec4d>             Vec4dVector;
  typedef VectorPOD<Vec4f>             Vec4fVector;
  typedef VectorPOD<Vec4i>             Vec4iVector;
  typedef VectorPOD<Vec4i8>            Vec4i8Vector;
  typedef VectorPOD<Vec4i16>           Vec4i16Vector;
  typedef VectorPOD<Vec4i32>           Vec4i32Vector;
  typedef VectorPOD<Vec4i64>           Vec4i64Vector;
  typedef VectorPOD<Vec4u>             Vec4uVector;
  typedef VectorPOD<Vec4u8>            Vec4u8Vector;
  typedef VectorPOD<Vec4u16>           Vec4u16Vector;
  typedef VectorPOD<Vec4u32>           Vec4u32Vector;
  typedef VectorPOD<Vec4u64>           Vec4u64Vector;

  typedef VectorPOD<Vec3>              Vec3Vector;
  typedef VectorPOD<Vec3d>             Vec3dVector;
  typedef VectorPOD<Vec3f>             Vec3fVector;
  typedef VectorPOD<Vec3i>             Vec3iVector;
  typedef VectorPOD<Vec3i8>            Vec3i8Vector;
  typedef VectorPOD<Vec3i16>           Vec3i16Vector;
  typedef VectorPOD<Vec3i32>           Vec3i32Vector;
  typedef VectorPOD<Vec3i64>           Vec3i64Vector;
  typedef VectorPOD<Vec3u>             Vec3uVector;
  typedef VectorPOD<Vec3u8>            Vec3u8Vector;
  typedef VectorPOD<Vec3u16>           Vec3u16Vector;
  typedef VectorPOD<Vec3u32>           Vec3u32Vector;
  typedef VectorPOD<Vec3u64>           Vec3u64Vector;

  typedef VectorPOD<Vec2>              Vec2Vector;
  typedef VectorPOD<Vec2d>             Vec2dVector;
  typedef VectorPOD<Vec2f>             Vec2fVector;
  typedef VectorPOD<Vec2i>             Vec2iVector;
  typedef VectorPOD<Vec2i8>            Vec2i8Vector;
  typedef VectorPOD<Vec2i16>           Vec2i16Vector;
  typedef VectorPOD<Vec2i32>           Vec2i32Vector;
  typedef VectorPOD<Vec2i64>           Vec2i64Vector;
  typedef VectorPOD<Vec2u>             Vec2uVector;
  typedef VectorPOD<Vec2u8>            Vec2u8Vector;
  typedef VectorPOD<Vec2u16>           Vec2u16Vector;
  typedef VectorPOD<Vec2u32>           Vec2u32Vector;
  typedef VectorPOD<Vec2u64>           Vec2u64Vector;

  typedef VectorPOD<SPDMatrix3x3>      SPDMatrix3x3Vector;
  typedef VectorPOD<Quat>              QuatVector;
  typedef VectorPOD<AffineMatrix4x4>   AffineMatrix4x4Vector;
  typedef VectorPOD<AffineMatrix4x4f>  AffineMatrix4x4fVector;
  typedef VectorPOD<AffineMatrix4x4d>  AffineMatrix4x4dVector;
  typedef VectorPOD<Bool>              BoolVector;
  typedef VectorPOD<char>              CharVector;


  typedef VectorPOD<ConstraintImplementation*>           ConstraintImplPtrVector;
  typedef VectorPOD<Attachment*>                         AttachmentPtrVector;

  typedef VectorPOD<Int>   IntVector;
  typedef VectorPOD<Int64> Int64Vector;
  typedef VectorPOD<Int32> Int32Vector;
  typedef VectorPOD<Int16> Int16Vector;
  typedef VectorPOD<Int8>  Int8Vector;

  typedef VectorPOD<UInt>    UIntVector;
  typedef VectorPOD<UInt64>  UInt64Vector;
  typedef VectorPOD<UInt32>  UInt32Vector;
  typedef VectorPOD<UInt16>  UInt16Vector;
  typedef VectorPOD<UInt8>   UInt8Vector;
  typedef VectorPOD<Index>   IndexVector;

  typedef VectorPOD<Real>    RealVector;
  typedef VectorPOD<Real64>  Real64Vector;
  typedef VectorPOD<Real32>  Real32Vector;

  typedef agx::VectorPOD<FluidContactPoint> FluidContactVector;
  typedef agx::Vector<agx::ref_ptr<Referenced> > RefVector;
}
#endif
