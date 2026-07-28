#include "NvInferRuntime.h"
#include "multiscaleDeformableAttnPluginLegacy.h"

#include <iostream>

namespace
{
constexpr char const* kPluginNamespace = "pose_custom";

class PoseMsDeformAttnRegistrar
{
public:
    PoseMsDeformAttnRegistrar()
    {
        mCreator.setPluginNamespace(kPluginNamespace);
        bool const ok = nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD)->registerCreator(mCreator, kPluginNamespace);
        if (!ok)
        {
            std::cerr << "[pose_msda_plugin] registerCreator failed for namespace " << kPluginNamespace << std::endl;
        }
    }

private:
    nvinfer1::plugin::MultiscaleDeformableAttnPluginCreatorLegacy mCreator{};
};

static PoseMsDeformAttnRegistrar gRegistrar{};
} // namespace
