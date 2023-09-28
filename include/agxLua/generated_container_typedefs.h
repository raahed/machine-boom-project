/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/


#ifndef AGXLUA_GENERATED_CONTAINER_TYPEDEFS_H
#define AGXLUA_GENERATED_CONTAINER_TYPEDEFS_H

#include <agx/Vec2.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>
#include <agx/Matrix3x3.h>
#include <agx/AffineMatrix4x4.h>
#include <agx/Quat.h>
#include <agx/Uuid.h>

namespace agx
{
  typedef agx::VectorPOD< agx::Real > RealVector;
  typedef agxData::Array< agx::Real > RealArray;
  typedef agxData::BufferT< agx::Real > RealBuffer;
  typedef agxData::ValueT< agx::Real > RealValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Real32 > Real32Vector;
  typedef agxData::Array< agx::Real32 > Real32Array;
  typedef agxData::BufferT< agx::Real32 > Real32Buffer;
  typedef agxData::ValueT< agx::Real32 > Real32Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Real64 > Real64Vector;
  typedef agxData::Array< agx::Real64 > Real64Array;
  typedef agxData::BufferT< agx::Real64 > Real64Buffer;
  typedef agxData::ValueT< agx::Real64 > Real64Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Bool > BoolVector;
  typedef agxData::Array< agx::Bool > BoolArray;
  typedef agxData::BufferT< agx::Bool > BoolBuffer;
  typedef agxData::ValueT< agx::Bool > BoolValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Index > IndexVector;
  typedef agxData::Array< agx::Index > IndexArray;
  typedef agxData::BufferT< agx::Index > IndexBuffer;
  typedef agxData::ValueT< agx::Index > IndexValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Int > IntVector;
  typedef agxData::Array< agx::Int > IntArray;
  typedef agxData::BufferT< agx::Int > IntBuffer;
  typedef agxData::ValueT< agx::Int > IntValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Int8 > Int8Vector;
  typedef agxData::Array< agx::Int8 > Int8Array;
  typedef agxData::BufferT< agx::Int8 > Int8Buffer;
  typedef agxData::ValueT< agx::Int8 > Int8Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Int16 > Int16Vector;
  typedef agxData::Array< agx::Int16 > Int16Array;
  typedef agxData::BufferT< agx::Int16 > Int16Buffer;
  typedef agxData::ValueT< agx::Int16 > Int16Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Int32 > Int32Vector;
  typedef agxData::Array< agx::Int32 > Int32Array;
  typedef agxData::BufferT< agx::Int32 > Int32Buffer;
  typedef agxData::ValueT< agx::Int32 > Int32Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Int64 > Int64Vector;
  typedef agxData::Array< agx::Int64 > Int64Array;
  typedef agxData::BufferT< agx::Int64 > Int64Buffer;
  typedef agxData::ValueT< agx::Int64 > Int64Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::UInt > UIntVector;
  typedef agxData::Array< agx::UInt > UIntArray;
  typedef agxData::BufferT< agx::UInt > UIntBuffer;
  typedef agxData::ValueT< agx::UInt > UIntValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::UInt8 > UInt8Vector;
  typedef agxData::Array< agx::UInt8 > UInt8Array;
  typedef agxData::BufferT< agx::UInt8 > UInt8Buffer;
  typedef agxData::ValueT< agx::UInt8 > UInt8Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::UInt16 > UInt16Vector;
  typedef agxData::Array< agx::UInt16 > UInt16Array;
  typedef agxData::BufferT< agx::UInt16 > UInt16Buffer;
  typedef agxData::ValueT< agx::UInt16 > UInt16Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::UInt32 > UInt32Vector;
  typedef agxData::Array< agx::UInt32 > UInt32Array;
  typedef agxData::BufferT< agx::UInt32 > UInt32Buffer;
  typedef agxData::ValueT< agx::UInt32 > UInt32Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::UInt64 > UInt64Vector;
  typedef agxData::Array< agx::UInt64 > UInt64Array;
  typedef agxData::BufferT< agx::UInt64 > UInt64Buffer;
  typedef agxData::ValueT< agx::UInt64 > UInt64Value;
}

namespace agx
{
  typedef agx::Vector< agx::String > StringVector;
  typedef agxData::Array< agx::String > StringArray;
  typedef agxData::BufferT< agx::String > StringBuffer;
  typedef agxData::ValueT< agx::String > StringValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec2 > Vec2Vector;
  typedef agxData::Array< agx::Vec2 > Vec2Array;
  typedef agxData::BufferT< agx::Vec2 > Vec2Buffer;
  typedef agxData::ValueT< agx::Vec2 > Vec2Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec2f > Vec2fVector;
  typedef agxData::Array< agx::Vec2f > Vec2fArray;
  typedef agxData::BufferT< agx::Vec2f > Vec2fBuffer;
  typedef agxData::ValueT< agx::Vec2f > Vec2fValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec2d > Vec2dVector;
  typedef agxData::Array< agx::Vec2d > Vec2dArray;
  typedef agxData::BufferT< agx::Vec2d > Vec2dBuffer;
  typedef agxData::ValueT< agx::Vec2d > Vec2dValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec2i > Vec2iVector;
  typedef agxData::Array< agx::Vec2i > Vec2iArray;
  typedef agxData::BufferT< agx::Vec2i > Vec2iBuffer;
  typedef agxData::ValueT< agx::Vec2i > Vec2iValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec2i32 > Vec2i32Vector;
  typedef agxData::Array< agx::Vec2i32 > Vec2i32Array;
  typedef agxData::BufferT< agx::Vec2i32 > Vec2i32Buffer;
  typedef agxData::ValueT< agx::Vec2i32 > Vec2i32Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec2i64 > Vec2i64Vector;
  typedef agxData::Array< agx::Vec2i64 > Vec2i64Array;
  typedef agxData::BufferT< agx::Vec2i64 > Vec2i64Buffer;
  typedef agxData::ValueT< agx::Vec2i64 > Vec2i64Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec2u > Vec2uVector;
  typedef agxData::Array< agx::Vec2u > Vec2uArray;
  typedef agxData::BufferT< agx::Vec2u > Vec2uBuffer;
  typedef agxData::ValueT< agx::Vec2u > Vec2uValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec2u32 > Vec2u32Vector;
  typedef agxData::Array< agx::Vec2u32 > Vec2u32Array;
  typedef agxData::BufferT< agx::Vec2u32 > Vec2u32Buffer;
  typedef agxData::ValueT< agx::Vec2u32 > Vec2u32Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec2u64 > Vec2u64Vector;
  typedef agxData::Array< agx::Vec2u64 > Vec2u64Array;
  typedef agxData::BufferT< agx::Vec2u64 > Vec2u64Buffer;
  typedef agxData::ValueT< agx::Vec2u64 > Vec2u64Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec3 > Vec3Vector;
  typedef agxData::Array< agx::Vec3 > Vec3Array;
  typedef agxData::BufferT< agx::Vec3 > Vec3Buffer;
  typedef agxData::ValueT< agx::Vec3 > Vec3Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec3f > Vec3fVector;
  typedef agxData::Array< agx::Vec3f > Vec3fArray;
  typedef agxData::BufferT< agx::Vec3f > Vec3fBuffer;
  typedef agxData::ValueT< agx::Vec3f > Vec3fValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec3d > Vec3dVector;
  typedef agxData::Array< agx::Vec3d > Vec3dArray;
  typedef agxData::BufferT< agx::Vec3d > Vec3dBuffer;
  typedef agxData::ValueT< agx::Vec3d > Vec3dValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec3i > Vec3iVector;
  typedef agxData::Array< agx::Vec3i > Vec3iArray;
  typedef agxData::BufferT< agx::Vec3i > Vec3iBuffer;
  typedef agxData::ValueT< agx::Vec3i > Vec3iValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec3i32 > Vec3i32Vector;
  typedef agxData::Array< agx::Vec3i32 > Vec3i32Array;
  typedef agxData::BufferT< agx::Vec3i32 > Vec3i32Buffer;
  typedef agxData::ValueT< agx::Vec3i32 > Vec3i32Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec3i64 > Vec3i64Vector;
  typedef agxData::Array< agx::Vec3i64 > Vec3i64Array;
  typedef agxData::BufferT< agx::Vec3i64 > Vec3i64Buffer;
  typedef agxData::ValueT< agx::Vec3i64 > Vec3i64Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec3u > Vec3uVector;
  typedef agxData::Array< agx::Vec3u > Vec3uArray;
  typedef agxData::BufferT< agx::Vec3u > Vec3uBuffer;
  typedef agxData::ValueT< agx::Vec3u > Vec3uValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec3u32 > Vec3u32Vector;
  typedef agxData::Array< agx::Vec3u32 > Vec3u32Array;
  typedef agxData::BufferT< agx::Vec3u32 > Vec3u32Buffer;
  typedef agxData::ValueT< agx::Vec3u32 > Vec3u32Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec3u64 > Vec3u64Vector;
  typedef agxData::Array< agx::Vec3u64 > Vec3u64Array;
  typedef agxData::BufferT< agx::Vec3u64 > Vec3u64Buffer;
  typedef agxData::ValueT< agx::Vec3u64 > Vec3u64Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec4 > Vec4Vector;
  typedef agxData::Array< agx::Vec4 > Vec4Array;
  typedef agxData::BufferT< agx::Vec4 > Vec4Buffer;
  typedef agxData::ValueT< agx::Vec4 > Vec4Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec4f > Vec4fVector;
  typedef agxData::Array< agx::Vec4f > Vec4fArray;
  typedef agxData::BufferT< agx::Vec4f > Vec4fBuffer;
  typedef agxData::ValueT< agx::Vec4f > Vec4fValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec4d > Vec4dVector;
  typedef agxData::Array< agx::Vec4d > Vec4dArray;
  typedef agxData::BufferT< agx::Vec4d > Vec4dBuffer;
  typedef agxData::ValueT< agx::Vec4d > Vec4dValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec4i > Vec4iVector;
  typedef agxData::Array< agx::Vec4i > Vec4iArray;
  typedef agxData::BufferT< agx::Vec4i > Vec4iBuffer;
  typedef agxData::ValueT< agx::Vec4i > Vec4iValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec4i32 > Vec4i32Vector;
  typedef agxData::Array< agx::Vec4i32 > Vec4i32Array;
  typedef agxData::BufferT< agx::Vec4i32 > Vec4i32Buffer;
  typedef agxData::ValueT< agx::Vec4i32 > Vec4i32Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec4i64 > Vec4i64Vector;
  typedef agxData::Array< agx::Vec4i64 > Vec4i64Array;
  typedef agxData::BufferT< agx::Vec4i64 > Vec4i64Buffer;
  typedef agxData::ValueT< agx::Vec4i64 > Vec4i64Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec4u > Vec4uVector;
  typedef agxData::Array< agx::Vec4u > Vec4uArray;
  typedef agxData::BufferT< agx::Vec4u > Vec4uBuffer;
  typedef agxData::ValueT< agx::Vec4u > Vec4uValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec4u32 > Vec4u32Vector;
  typedef agxData::Array< agx::Vec4u32 > Vec4u32Array;
  typedef agxData::BufferT< agx::Vec4u32 > Vec4u32Buffer;
  typedef agxData::ValueT< agx::Vec4u32 > Vec4u32Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Vec4u64 > Vec4u64Vector;
  typedef agxData::Array< agx::Vec4u64 > Vec4u64Array;
  typedef agxData::BufferT< agx::Vec4u64 > Vec4u64Buffer;
  typedef agxData::ValueT< agx::Vec4u64 > Vec4u64Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Matrix3x3 > Matrix3x3Vector;
  typedef agxData::Array< agx::Matrix3x3 > Matrix3x3Array;
  typedef agxData::BufferT< agx::Matrix3x3 > Matrix3x3Buffer;
  typedef agxData::ValueT< agx::Matrix3x3 > Matrix3x3Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Matrix3x3f > Matrix3x3fVector;
  typedef agxData::Array< agx::Matrix3x3f > Matrix3x3fArray;
  typedef agxData::BufferT< agx::Matrix3x3f > Matrix3x3fBuffer;
  typedef agxData::ValueT< agx::Matrix3x3f > Matrix3x3fValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Matrix3x3d > Matrix3x3dVector;
  typedef agxData::Array< agx::Matrix3x3d > Matrix3x3dArray;
  typedef agxData::BufferT< agx::Matrix3x3d > Matrix3x3dBuffer;
  typedef agxData::ValueT< agx::Matrix3x3d > Matrix3x3dValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Matrix4x4 > Matrix4x4Vector;
  typedef agxData::Array< agx::Matrix4x4 > Matrix4x4Array;
  typedef agxData::BufferT< agx::Matrix4x4 > Matrix4x4Buffer;
  typedef agxData::ValueT< agx::Matrix4x4 > Matrix4x4Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Matrix4x4f > Matrix4x4fVector;
  typedef agxData::Array< agx::Matrix4x4f > Matrix4x4fArray;
  typedef agxData::BufferT< agx::Matrix4x4f > Matrix4x4fBuffer;
  typedef agxData::ValueT< agx::Matrix4x4f > Matrix4x4fValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Matrix4x4d > Matrix4x4dVector;
  typedef agxData::Array< agx::Matrix4x4d > Matrix4x4dArray;
  typedef agxData::BufferT< agx::Matrix4x4d > Matrix4x4dBuffer;
  typedef agxData::ValueT< agx::Matrix4x4d > Matrix4x4dValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::AffineMatrix4x4 > AffineMatrix4x4Vector;
  typedef agxData::Array< agx::AffineMatrix4x4 > AffineMatrix4x4Array;
  typedef agxData::BufferT< agx::AffineMatrix4x4 > AffineMatrix4x4Buffer;
  typedef agxData::ValueT< agx::AffineMatrix4x4 > AffineMatrix4x4Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::AffineMatrix4x4f > AffineMatrix4x4fVector;
  typedef agxData::Array< agx::AffineMatrix4x4f > AffineMatrix4x4fArray;
  typedef agxData::BufferT< agx::AffineMatrix4x4f > AffineMatrix4x4fBuffer;
  typedef agxData::ValueT< agx::AffineMatrix4x4f > AffineMatrix4x4fValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::AffineMatrix4x4d > AffineMatrix4x4dVector;
  typedef agxData::Array< agx::AffineMatrix4x4d > AffineMatrix4x4dArray;
  typedef agxData::BufferT< agx::AffineMatrix4x4d > AffineMatrix4x4dBuffer;
  typedef agxData::ValueT< agx::AffineMatrix4x4d > AffineMatrix4x4dValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Quat > QuatVector;
  typedef agxData::Array< agx::Quat > QuatArray;
  typedef agxData::BufferT< agx::Quat > QuatBuffer;
  typedef agxData::ValueT< agx::Quat > QuatValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Quat32 > Quat32Vector;
  typedef agxData::Array< agx::Quat32 > Quat32Array;
  typedef agxData::BufferT< agx::Quat32 > Quat32Buffer;
  typedef agxData::ValueT< agx::Quat32 > Quat32Value;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Quat64 > Quat64Vector;
  typedef agxData::Array< agx::Quat64 > Quat64Array;
  typedef agxData::BufferT< agx::Quat64 > Quat64Buffer;
  typedef agxData::ValueT< agx::Quat64 > Quat64Value;
}

namespace agxData
{
  typedef agx::VectorPOD< agxData::EntityPtr > EntityPtrVector;
  typedef agxData::Array< agxData::EntityPtr > EntityPtrArray;
  typedef agxData::BufferT< agxData::EntityPtr > EntityPtrBuffer;
  typedef agxData::ValueT< agxData::EntityPtr > EntityPtrValue;
}

namespace agx
{
  typedef agx::VectorPOD< agx::Uuid > UuidVector;
  typedef agxData::Array< agx::Uuid > UuidArray;
  typedef agxData::BufferT< agx::Uuid > UuidBuffer;
  typedef agxData::ValueT< agx::Uuid > UuidValue;
}


#endif //AGXLUA_GENERATED_CONTAINER_TYPEDEFS_H
