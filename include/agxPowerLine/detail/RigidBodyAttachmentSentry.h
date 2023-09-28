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


#ifndef AGXPOWERLINE_DETAIL_RIGID_BODY_ATTACHMENT_H
#define AGXPOWERLINE_DETAIL_RIGID_BODY_ATTACHMENT_H

#include <agxStream/Serializable.h>

namespace agxPowerLine
{
  namespace detail
  {
    /**
    RigidBodyTranslationalAttachment has been deprecated and removed. This sentry
    class detects the presense of such objects in old archive serializations
    and uses LOGGER_ERROR to abort the restore process.
    */
    class RigidBodyTranslationalAttachmentSentry : public agxStream::Serializable
    {
      public:
        virtual void store(agxStream::OutputArchive& out) const override;
        virtual void restore(agxStream::InputArchive& in) override;

        virtual agxStream::StorageAgent* getStorageAgent() const override;
        static const char* getConstructClassId();
        friend class agxStream::DefStorageAgent<agxPowerLine::detail::RigidBodyTranslationalAttachmentSentry>;
        virtual const char* getClassName() const override;
        static agxStream::Serializable* create();
        static agxStream::Serializable* create(agxStream::InputArchive&);

      private:
        RigidBodyTranslationalAttachmentSentry();
        virtual ~RigidBodyTranslationalAttachmentSentry();
    };


    /**
    RigidBodyTranslationalAttachment has been deprecated and removed. This sentry
    class detects the presence of such objects in old archive serializations
    and uses LOGGER_ERROR to abort the restore process.
    */
    class RigidBodyRotationalAttachmentSentry : public agxStream::Serializable
    {
      public:
        virtual void store(agxStream::OutputArchive& out) const override;
        virtual void restore(agxStream::InputArchive& in) override;

        virtual agxStream::StorageAgent* getStorageAgent() const override;
        static const char* getConstructClassId();
        friend class agxStream::DefStorageAgent<agxPowerLine::detail::RigidBodyRotationalAttachmentSentry>;
        virtual const char* getClassName() const override;
        static agxStream::Serializable* create();
        static agxStream::Serializable* create(agxStream::InputArchive&);

      private:
        RigidBodyRotationalAttachmentSentry();
        virtual ~RigidBodyRotationalAttachmentSentry();
    };
  }
}

#endif
