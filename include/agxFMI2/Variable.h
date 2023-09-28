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

#ifndef AGXFMI2_VARIABLE_H
#define AGXFMI2_VARIABLE_H

#include <agx/config/AGX_USE_FMI.h>

#if AGX_USE_FMI()

#include <agxFMI2/export.h>
#include <agx/Object.h>

namespace agxFMI2
{
  AGX_DECLARE_POINTER_TYPES(Variable);
  AGX_DECLARE_VECTOR_TYPES(Variable);

  /**
  FMI variable wrapper. See FMI specification document for further details.
  */
  class AGXFMI_EXPORT Variable : public agx::Object
  {
  public:

    enum Type
    {
      TYPE_REAL,
      TYPE_INT,
      TYPE_BOOL,
      TYPE_STRING,
      TYPE_ENUM,
      TYPE_UNKNOWN
    };

    enum Variability
    {
      VARIABILITY_CONSTANT,
      VARIABILITY_FIXED,
      VARIABILITY_TUNABLE,
      VARIABILITY_DISCRETE,
      VARIABILITY_CONTINOUS,
      VARIABILITY_UNKNOWN
    };

    enum Causality
    {
      CAUSALITY_PARAMETER,
      CAUSALITY_CALCULATED_PARAMETER,
      CAUSALITY_INPUT,
      CAUSALITY_OUTPUT,
      CAUSALITY_LOCAL,
      CAUSALITY_INDEPENDENT,
      CAUSALITY_UNKNOWN
    };

    enum Initiality
    {
      INITIALITY_EXACT,
      INITIALITY_APPROX,
      INITIALITY_CALCULATED,
      INITIALITY_UNKNOWN
    };


  public:
    Variable(const agx::Name& name = agx::Name());

    /// Set the description.
    void setDescription(const agx::String& description);

    /// \return The description
    const agx::String& getDescription() const;

    /// Set the value reference
    void setValueReference(agx::UInt32 vr);

    /// \return The value reference
    agx::UInt32 getValueReference() const;

    /// Set the type, automatically set by child classes
    void setType(Type type);

    /// \return The type
    Type getType() const;

    /// \return The type as a string
    const char *getTypeString();

    /// Set the variability
    void setVariability(Variability variability);

    /// \return The variability
    Variability getVariability() const;

    /// \return The variability as a string
    const char *getVariabilityString();

    /// Set the causality
    void setCausality(Causality causality);

    /// \return The causality
    Causality getCausality() const;

    /// \return The causality as a string
    const char *getCausalityString();


    /// Set the initiality
    void setInitiality(Initiality initiality);

    /// \return The initiality
    Initiality getInitiality() const;

    /// \return The initiality as a string
    const char *getInitialityString() const;

  protected:
    virtual ~Variable();

  protected:
    agx::String m_description;
    agx::UInt32 m_valueReference;
    Type m_type;
    Variability m_variability;
    Causality m_causality;
    Initiality m_initiality;
  };


}

#endif /* AGX_USE_FMI */

#endif /* AGXFMI2_VARIABLE_H */
