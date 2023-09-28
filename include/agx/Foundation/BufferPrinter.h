/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_BUFFERPRINTER_H
#define AGX_BUFFERPRINTER_H

#include <agx/Task.h>
#include <agx/TaskTemplate.h>
#include <agx/IndexRange.h>
#include <agx/Parameter.h>
#include <agx/Vector.h>
#include <agxIO/Events.h>
#include <agx/Bound.h>
#include <agx/String.h>

namespace agx
{
  class BufferPrinter : public SerialTask
  {
  public:
    BufferPrinter(Device* device);
    ~BufferPrinter();
    virtual void run();

  private:
    ScalarParameterRef m_rangeParameter;
    agx::Bound1u m_range;
    size_t m_rangeLength;
    agxIO::InputEvent::CallbackType m_printAlwaysCallback;
    agxIO::InputEvent::CallbackType m_printOnceCallback;
    agxIO::InputEvent::CallbackType m_printImmediatelyCallback;

    void print();
    void requestOnePrint();
    void requestContinuousPrint();
    void requestImmediatePrint();
    bool m_printOnce;

    typedef Vector<agx::String> ColumnContent;
    typedef Vector<ColumnContent> Columns;
    Columns m_toPrint;
    Vector<agx::UInt> m_columnWidths;

    void fillToPrint(const ParameterPtrVector& arguments);
    void findColumnWidths();
    void printColumns();

  };
}

#endif

