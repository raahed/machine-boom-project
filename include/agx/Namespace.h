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

#ifndef AGX_NAMESPACE_H
#define AGX_NAMESPACE_H

#include <agx/Component.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES(Namespace);

  /**
  A namespace is used as a group node in the agx::Model hierarchy.
  */
  class AGXCORE_EXPORT Namespace : public Component
  {
  public:
    static agx::Model *ClassModel();

    Namespace(const agx::Name& name, Model *model = Namespace::ClassModel());

    virtual String autoComplete(const String& partialName, StringVector& matchingNames) const override;

  protected:
    virtual ~Namespace();

  private:
  };

}


#endif /* AGX_NAMESPACE_H */
