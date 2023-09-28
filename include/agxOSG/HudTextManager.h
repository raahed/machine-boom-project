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

#ifndef AGXOSG_HUDTEXTMANAGER_H
#define AGXOSG_HUDTEXTMANAGER_H

#include <agxOSG/export.h>
#include <agxOSG/SceneDecorator.h>
#include <agxSDK/StepEventListener.h>


/// Namespace holding utility classes and functions for connecting AGX to the OpenSceneGraph rendering API.
namespace agxOSG
{


  AGX_DECLARE_POINTER_TYPES(HudTextManager);

  /**
  HudTextManager is a class that inherits from agxSDK::StepEventListener and makes it easier
  to manipulate text in an agxOSG::SceneDecorator.
  Note that it will potentially reset all text in the SceneDecorator;
  mixing the use of HudTextManager and SceneDecorator::setText(...) is therefore not recommended.
  */
  class AGXOSG_EXPORT HudTextManager : public agxSDK::StepEventListener
  {

    public:
      AGX_DECLARE_POINTER_TYPES( ReferencedText );
      /// A class for having text elements with reference counting.
      class AGXOSG_EXPORT ReferencedText : public agx::Referenced
      {
        public:
          /// Creates a referenced text given a string.
          ReferencedText(const agx::String& text);

          /// Get the current text.
          const agx::String& getText() const;

          /// Set the new text.
          void setText(const agx::String& text);

        protected:
          virtual ~ReferencedText();

        protected:
          agx::String m_text;
      };

      HudTextManager(agxOSG::SceneDecorator* sceneDecorator);

      /// Add text which should be shown only this time step.
      void addText(const agx::String& text);

      /**
      Add text which should be shown all time steps (until removed).
      Constant text will be shown before normal text.
      */
      void addConstantText(const agx::String& text);

      /// Clear constant text.
      void clearConstantText();

      /// Clear text which should be shown only this time step.
      void clearText();

      /// Updates the HUD given constant text, referenced text and text, without clearing text.
      void updateHud();

      /**
      Adds referenced text. HudTextManager will only hold a weak reference,
      the caller should hold a strong one.
      As soon as the text gets deallocated, it will not be shown anymore.
      */
      void addReferencedText(ReferencedText* text);

      /// Inherited from agxSDK::StepEventListener.
      virtual void last(const agx::TimeStamp& t) override;

    protected:
      virtual ~HudTextManager();


    protected:
      osg::observer_ptr<agxOSG::SceneDecorator> m_sceneDecorator;
      agx::String m_text;
      agx::String m_constantText;
      agx::Vector<ReferencedTextObserver> m_referencedTexts;
  };

}



#endif


