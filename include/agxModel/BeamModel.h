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

#include <agxModel/BeamModelProperties.h>
#include <agxModel/BeamStiffnessDamping.h>

#include <agxCollide/Geometry.h>

namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES( BeamModel );
  AGX_DECLARE_POINTER_TYPES( IBeam );
  AGX_DECLARE_POINTER_TYPES( RectangularBeam );
  AGX_DECLARE_POINTER_TYPES( CircularBeam );
  AGX_DECLARE_POINTER_TYPES( HollowRectangularBeam );
  AGX_DECLARE_POINTER_TYPES( HollowCircularBeam );
  AGX_DECLARE_POINTER_TYPES( GenericBeamModel );

  /**
  Beam model interface requiring moment of inertia (area moment in x, y and polar moment in z),
  the cross section area, 2D (x and y) extents and geometry for the segments. The geometry
  is cloned to each created beam segment, i.e., called once when the beam is initialized.

  The moment of inertia is area moment (x, y) and polar moment (z) is defined about the local
  frame of the beam segments. Beam segments has x up, y right and z forward/along the beam:

      x
      |  z
      | /
      |/____ y

  I.e., lateral moment is x, vertical moment is y and polar moment is z. Note that resources
  normally has x right and y up.

  The geometry must be centered in the local frame with extents segmentLength/2 forward and
  back. The height is along x, width along y and length along z. E.g., a rectangular model
  with given width and height should be created as:

      agxCollide::GeometryRef geometry = new agxCollide::Geometry( new agxCollide::Box( 0.5 * height,
                                                                                        0.5 * width,
                                                                                        0.5 * segmentLength ) );

  with moment of inertia:

      (x: height * width^3 / 12,
       y: width * height^3 / 12,
       z: width * height * (width^2 + height^2) / 12)
  */
  class AGXMODEL_EXPORT BeamModel : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Assign properties instance, a new default will be created if \p properties
      is nullptr. The properties instance may be shared between an arbitrary
      number of beam models.
      \param properties - new beam properties instance
      */
      void setProperties( BeamModelProperties* properties );

      /**
      \return properties of the model
      */
      BeamModelProperties* getProperties() const;

      /**
      Calculates stiffness dependent on moment of inertia, cross section area, beam segment
      length and number of segments in the beam.
      \note The arguments aren't validated, what you give is what you get, i.e., assert that
            \p numBeamSegments > 0 and \p beamSegmentLength > 0.0.
      \param numBeamSegments - total number of beam segments in the beam this model belongs to
      \param beamSegmentLength - length of a beam segment, i.e., total length of the beam is beamSegmentLength * numBeamSegments
      \return the calculated translational and rotational stiffness and damping times
      */
      BeamStiffnessDamping calculateStiffnessDamping( agx::UInt numBeamSegments,
                                                      agx::Real beamSegmentLength ) const;

      /**
      Scale of the calculated moment of inertia of this model, default scale (1, 1, 1).

      The moment of inertia, specifically the polar moment, is not accurate and not possible
      to calculate analytically for beam models that hasn't a circular cross section. This
      scale enables modifications of the moment of inertia for specific dimensions of this beam
      model. E.g., alternate the scale until the resulting twist angle, under given constant
      torque, matches the angle given in some resource for an identical beam model.

      x: About up/down, lateral.
      y: About right/left, horizontal.
      z: About forward/backward, torsional/polar.

      \param momentOfInertiaScale - scale of this model specific moment of inertia (default scale: (1, 1, 1))
      */
      void setMomentOfInertiaScale( agx::Vec3 momentOfInertiaScale );

      /**
      \return the currently used moment of inertia scale (default scale: (1, 1, 1))
      */
      agx::Vec3 getMomentOfInertiaScale() const;

      /**
      Assign new polar moment of inertia scale.
      \sa setMomentOfInertiaScale
      \param polarMomentOfInertiaScale - new polar moment of inertia (default: 1)
      */
      void setPolarMomentOfInertiaScale( agx::Real polarMomentOfInertiaScale );

      /**
      \return the current polar moment of inertia
      */
      agx::Real getPolarMomentOfInertiaScale() const;

    public:
      /**
      Moment of inertia of the beam model, where:
          About x (up):      Lateral
          About y (right):   Vertical
          About z (forward): Torsional
      This is the moment of inertia used to calculate the stiffness in the constraints of
      the beam, and is scaled with the current moment of inertia scale (default scale: (1, 1, 1)).
      \return the scaled moment of inertia (x and y) and polar (z) of the beam model
      */
      virtual agx::Vec3 getMomentOfInertia() const final;

      /**
      Default moment of inertia of the beam model. This moment of inertia isn't scaled and
      is unique to the implementation. This method is normally only used by getMomentOfInertia.
          About x (up):      Lateral
          About y (right):   Vertical
          About z (forward): Torsional
      \return the unscaled moment of inertia of the beam model
      */
      virtual agx::Vec3 calculateMomentOfInertia() const = 0;

      /**
      The cross section area of the beam model affecting the stretch stiffness.
      \return the cross section area of the beam model
      */
      virtual agx::Real getCrossSectionArea() const = 0;

      /**
      The 2D extents along x (up/down) and y (right/left).
      \return the 2D extents of the beam model
      */
      virtual agx::Vec2 getMaxHeightWidth() const = 0;

      /**
      Generates a geometry to be cloned to each beam segment. This geometry must be centered
      in the local segment frame with x up, y right and z forward. The geometry
      is cloned for each segment and translated 0.5 * segmentLength forward.
      \param segmentLength - the total length of the segment, depending on the total
                             length and resolution of the beam which are unknown to
                             the beam model.
      \return geometry instance to be cloned to each beam segment
      */
      virtual agxCollide::GeometryRef getSegmentGeometry( agx::Real segmentLength ) const = 0;

      /**
      Validate inputs, verify all given sizes are > 0, e.g., hollow cylinder thickness < radius,
      or I-beam web thickness < width. Return the error, if any, or an empty string if the
      beam can be initialized given this model.
      \return empty string if valid, otherwise error
      */
      virtual agx::String validate() const = 0;


      /**
      Load named BeamModelProperties from the Material Library for the current Beam.
      \returns true if properties could be read successfully.
      */
      bool loadLibraryProperties(const agx::String& name);


      /**
      Returns a vector with all the named BeamModelProperties that were found in the Material Library
      \returns A vector with the names of all BeamModelProperties found in the MaterialLibrary
      */
      static agx::StringVector getAvailableLibraryProperties();



    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxModel::BeamModel );

      virtual void store( agxStream::OutputArchive& out ) const override;
      virtual void restore( agxStream::InputArchive& in ) override;

    protected:
      BeamModel();
      BeamModel( const BeamModel& ) = delete;
      BeamModel( BeamModelProperties* properties );
      virtual ~BeamModel();

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      BeamModelPropertiesRef m_properties;
      agx::Vec3 m_momentOfInertiaScale;
  };

  /**
  I-beam model with a given width, height, flange thickness and web thickness.

              ____________________________________________
      ^  ^   |                                            |
      |  |-C |                                            |
      |  v   |__________________        __________________|
      |                        |       |
      |                        |       |                     x
      |                        |   D   |                     |  z
      |-B                      |<----->|                     | /
      |                        |       |                     |/____ y
      |                        |       |
      |       _________________|       |__________________
      |  ^   |                                            |
      |  |-C |                                            |
      v  v   |____________________________________________|
                                   A
             <-------------------------------------------->

  A: Width - the total width of the beam.
  B: Height - the total height of the beam.
  C: Flange thickness - the thickness of the upper and lower flanges.
  D: Web thickness - the thickness of the web (middle column).
  */
  class AGXMODEL_EXPORT IBeam : public BeamModel
  {
    public:
      /**
      Construct an I-beam model given width, height, flange thickness, web thickness and
      optional properties. If \p properties is nullptr a new, default, beam model properties
      will be created.
      \param width - total width of the beam
      \param height - total height of the beam
      \param flangeThickness - thickness of the top and bottom flanges
      \param webThickness - thickness of the web
      \param properties - optional beam model properties, new default instance will be created if nullptr
      */
      IBeam( agx::Real width,
             agx::Real height,
             agx::Real flangeThickness,
             agx::Real webThickness,
             BeamModelProperties* properties = nullptr );

      /**
      \return the total width of the I-beam
      */
      agx::Real getWidth() const;

      /**
      \return the total height of the I-beam
      */
      agx::Real getHeight() const;

      /**
      \return the thickness of the flanges
      */
      agx::Real getFlangeThickness() const;

      /**
      \return the thickness of the web
      */
      agx::Real getWebThickness() const;

      /**
      \return the height of the web (inner height)
      */
      agx::Real getWebHeight() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::IBeam );

      /**
      Moment of inertia: (
          x: flangeThickness^3 * webHeight / 12 + width^3 * flangeThickness / 6,
          y: webHeight^3 * flangeThickness / 12 + height^3 * width / 12 - webHeight^3,
          z: x + y
      )
      */
      virtual agx::Vec3 calculateMomentOfInertia() const override;

      /**
      Cross section area: 2 * flangeThickness * width + webThickness * webHeight
      */
      virtual agx::Real getCrossSectionArea() const override;

      /**
      Maximum height and width: (height, width)
      */
      virtual agx::Vec2 getMaxHeightWidth() const override;

      /**
      Creates segment geometry of given length containing three boxes (flanges and web).
      */
      virtual agxCollide::GeometryRef getSegmentGeometry( agx::Real segmentLength ) const override;

      /**
      Validates: Width > 0
                 Height > 0
                 Flange thickness > 0 and 2.0 * Flange thickness = Web height < Height
                 Web thickness > 0 and Web thickness < Width
      */
      virtual agx::String validate() const override;

    protected:
      IBeam();
      virtual ~IBeam();

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::Real m_width;
      agx::Real m_height;
      agx::Real m_flangeThickness;
      agx::Real m_webThickness;
  };

  /**
  Rectangular cross section beam model with a given width and height.
           ____________________________________________
      ^   |                                            |
      |   |                                            |
      |   |                                            |
      |   |                                            |
      |   |                                            |     x
      |   |                                            |     |  z
      |-B |                                            |     | /
      |   |                                            |     |/____ y
      |   |                                            |
      |   |                                            |
      |   |                                            |
      |   |                                            |
      v   |____________________________________________|
                                A
          <-------------------------------------------->

  A: Width - the total width of the beam.
  B: Height - the total height of the beam.
  */
  class AGXMODEL_EXPORT RectangularBeam : public BeamModel
  {
    public:
      /**
      Construct a rectangular beam model with a given width, height and optional
      beam model properties. If \p properties is nullptr a new, default, beam model
      properties will be created.
      \param width - total width of the rectangular beam
      \param height - total height of the rectangular beam
      \param properties - optional beam model properties, new default instance will be created if nullptr
      */
      RectangularBeam( agx::Real width, agx::Real height, BeamModelProperties* properties = nullptr );

      /**
      \return the total width of this rectangular beam model
      */
      agx::Real getWidth() const;

      /**
      \return the total height of this rectangular beam model
      */
      agx::Real getHeight() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::RectangularBeam );

      /**
      Moment of inertia: (
          x: width^3 * height / 12,
          y: height^3 * width / 12,
          z: a = 0.5 * max( width, height )
             b = 0.5 * min( width, height )
             result = a * b^3 * (16 / 3 - 3.36 * (a / b) * (1.0 - b^4 / (12 * a^4))
      )
      */
      virtual agx::Vec3 calculateMomentOfInertia() const override;

      /**
      Cross section area: width * height
      */
      virtual agx::Real getCrossSectionArea() const override;

      /**
      Maximum height and width: (height, width)
      */
      virtual agx::Vec2 getMaxHeightWidth() const override;

      /**
      Creates segment geometry of given length containing one box with half extents:
          (0.5 * height, 0.5 * width, 0.5 * segmentLength)
      */
      virtual agxCollide::GeometryRef getSegmentGeometry( agx::Real segmentLength ) const override;

      /**
      Validates: Width > 0
                 Height > 0
      */
      virtual agx::String validate() const override;

    protected:
      RectangularBeam();
      virtual ~RectangularBeam();

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::Real m_width;
      agx::Real m_height;
  };

  /**
  Circular cross section beam model with a given radius.
               , - ~ ~ ~ - ,
           , '               ' ,
         ,                       ,
        ,                         ,
       ,       A                   ,                         x
       ,<----------->              ,                         |  z
       ,                           ,                         | /
        ,                         ,                          |/____ y
         ,                       ,
           ,                  , '
             ' - , _ _ _ ,  '

  A: Radius - radius of the beam.
  */
  class AGXMODEL_EXPORT CircularBeam : public BeamModel
  {
    public:
      /**
      Construct a circular cross section beam model with a given radius and optional
      beam model properties. If \p properties is nullptr a new, default, beam model
      properties will be created.
      \param radius - radius of the circular beam
      \param properties - optional beam model properties, new default instance will be created if nullptr
      */
      CircularBeam( agx::Real radius, BeamModelProperties* properties = nullptr );

      /**
      \return the radius of this circular beam model
      */
      agx::Real getRadius() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::CircularBeam );

      /**
      Moment of inertia: (
          x: pi * radius^4 / 4,
          y: x,
          z: x + y
      )
      */
      virtual agx::Vec3 calculateMomentOfInertia() const override;

      /**
      Cross section area: pi * radius^2
      */
      virtual agx::Real getCrossSectionArea() const override;

      /**
      Max height and width: (2 * radius, 2 * radius)
      */
      virtual agx::Vec2 getMaxHeightWidth() const override;

      /**
      Creates segment geometry of given length containing one cylinder.
      */
      virtual agxCollide::GeometryRef getSegmentGeometry( agx::Real segmentLength ) const override;

      /**
      Validates: Radius > 0
      */
      virtual agx::String validate() const override;

    protected:
      CircularBeam();
      virtual ~CircularBeam();

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::Real m_radius;
  };

  /**
  Hollow rectangular cross section beam model with a given width, height and thickness.
           ____________________________________________  v
      ^   |                                            | |-C
      |   |      ________________________________      | |
      |   |     |                                |     | ^
      |   |     |                                |     |
      |   |     |                                |     |     x
      |   |  C  |                                |  C  |     |  z
      |-B |<--->|                                |<--->|     | /
      |   |     |                                |     |     |/____ y
      |   |     |                                |     |
      |   |     |                                |     |
      |   |     |________________________________|     | v
      |   |                                            | |-C
      v   |____________________________________________| |
                               A                         ^
          <-------------------------------------------->

  A: Width - the total width of the beam.
  B: Height - the total height of the beam.
  C: Thickness - the overall thickness of the beam.
  */
  class AGXMODEL_EXPORT HollowRectangularBeam : public BeamModel
  {
    public:
      /**
      Construct a hollow rectangular cross section beam model given width,
      height, thickness and optional beam model properties. The hollow
      width will become width - 2 * thickness and the hollow height will
      become height - 2 * thickness. If \p properties is nullptr a new,
      default, beam model properties will be created.
      \param width - total width of the rectangular beam
      \param height - total height of the rectangular beam
      \param thickness - overall thickness of the rectangular beam
      \param properties - optional beam model properties, new default instance will be created if nullptr
      */
      HollowRectangularBeam( agx::Real width,
                             agx::Real height,
                             agx::Real thickness,
                             BeamModelProperties* properties = nullptr );

      /**
      \return the width of this hollow rectangular beam model
      */
      agx::Real getWidth() const;

      /**
      \return the width of the hollow of this rectangular beam model
      */
      agx::Real getHollowWidth() const;

      /**
      \return the height of this hollow rectangular beam model
      */
      agx::Real getHeight() const;

      /**
      \return the height of the hollow of this rectangular beam model
      */
      agx::Real getHollowHeight() const;

      /**
      \return the overall thickness of this rectangular beam model
      */
      agx::Real getThickness() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::HollowRectangularBeam );

      /**
      Moment of inertia: (
          x: (width^3 * height - hollowWidth^3 * hollowHeight) / 16,
          y: (height^3 * width - hollowHeight^3 * hollowWidth) / 16,
          z: x + y
      )
      */
      virtual agx::Vec3 calculateMomentOfInertia() const override;

      /**
      Cross section area: width * height - hollowWidth * hollowHeight
      */
      virtual agx::Real getCrossSectionArea() const override;

      /**
      Max height and width: (height, width)
      */
      virtual agx::Vec2 getMaxHeightWidth() const override;

      /**
      Creates segment geometry of given length containing four boxes.
      */
      virtual agxCollide::GeometryRef getSegmentGeometry( agx::Real segmentLength ) const override;

      /**
      Validates: Width > 0
                 Height > 0
                 Thickness > 0 and 2 * Thickness < Width and 2 * Thickness < Height
      */
      virtual agx::String validate() const override;

    protected:
      HollowRectangularBeam();
      virtual ~HollowRectangularBeam();

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::Real m_width;
      agx::Real m_height;
      agx::Real m_thickness;
  };

  /**
  Hollow circular cross section beam model with a given (outer) radius and thickness.
  */
  class AGXMODEL_EXPORT HollowCircularBeam : public BeamModel
  {
    public:
      /**
      Construct a hollow circular cross section beam model given (outer) radius and
      thickness. The hollow radius will become radius - thickness. If \p properties
      is nullptr a new, default, beam model properties will be created.
      \param radius - the total radius of the beam
      \param thickness - the thickness of the beam, resulting in the hollow radius: radius - thickness
      \param properties - optional beam model properties, new default instance will be created if nullptr
      */
      HollowCircularBeam( agx::Real radius,
                          agx::Real thickness,
                          BeamModelProperties* properties = nullptr );

      /**
      \return the (outer/total) radius of this hollow circular beam model
      */
      agx::Real getRadius() const;

      /**
      \return the thickness of this hollow circular beam model
      */
      agx::Real getThickness() const;

      /**
      \return the radius of the hollow of this circular beam model
      */
      agx::Real getHollowRadius() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::HollowCircularBeam );

      /**
      Moment of inertia: (
          x: pi * (radius^4 - hollowRadius^4) / 4,
          y: x,
          z: x + y
      )
      */
      virtual agx::Vec3 calculateMomentOfInertia() const override;

      /**
      Cross section area: pi * (radius^2 - hollowRadius^2)
      */
      virtual agx::Real getCrossSectionArea() const override;

      /**
      Max height and width: (2 * radius, 2 * radius)
      */
      virtual agx::Vec2 getMaxHeightWidth() const override;

      /**
      Creates segment geometry of given length containing one hollow cylinder.
      */
      virtual agxCollide::GeometryRef getSegmentGeometry( agx::Real segmentLength ) const override;

      /**
      Validates: Radius > 0
                 Thickness > 0 and Thickness < Radius
      */
      virtual agx::String validate() const override;

    protected:
      HollowCircularBeam();
      virtual ~HollowCircularBeam();

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::Real m_radius;
      agx::Real m_thickness;
  };

  /**
  Generic beam model throwing exceptions for non-implemented virtual methods.
  This model is the model to inherit from in, e.g, Python or C#.
  */
  class AGXMODEL_EXPORT GenericBeamModel : public BeamModel
  {
    public:
      /**
      Construct given optional model properties. If \p properties is nullptr a new,
      default, beam model properties will be created.
      */
      GenericBeamModel( BeamModelProperties* properties = nullptr );

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      /**
      Calculate the moment of inertia of this model. GenericBeamModel::getMomentOfInertia
      will throw an error if called - this method has to be overridden.
      */
      virtual agx::Vec3 calculateMomentOfInertia() const override;

      /**
      Calculate the cross section area of this model. GenericBeamModel::getCrossSectionArea
      will throw an error if called - this method has to be overridden.
      */
      virtual agx::Real getCrossSectionArea() const override;

      /**
      Calculate the max height and width of this model. GenericBeamModel::getMaxHeightWidth
      will throw an error if called - this method has to be overridden.
      */
      virtual agx::Vec2 getMaxHeightWidth() const override;

      /**
      Create segment geometry of this model. GenericBeamModel::getSegmentGeometry
      will throw an error if called - this method has to be overridden.
      */
      virtual agxCollide::GeometryRef getSegmentGeometry( agx::Real segmentLength ) const override;

      /**
      Validate inputs to this model. GenericBeamModel::validate
      will throw an error if called - this method has to be overridden.
      */
      virtual agx::String validate() const override;

    protected:
      virtual ~GenericBeamModel();

      DOXYGEN_END_INTERNAL_BLOCK()
  };
}
