#include "common/plugin.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace nvinfer1::plugin
{

ILogger* gLogger{};

template <ILogger::Severity tSeverity>
int32_t LogStream<tSeverity>::Buf::sync()
{
    std::string s = str();
    while (!s.empty() && s.back() == '\n')
    {
        s.pop_back();
    }
    if (!s.empty())
    {
        if (gLogger != nullptr)
        {
            gLogger->log(tSeverity, s.c_str());
        }
        else
        {
            std::fprintf(stderr, "[pose TRT plugin] %s\n", s.c_str());
        }
    }
    str("");
    return 0;
}

LogStream<ILogger::Severity::kERROR> gLogError;
LogStream<ILogger::Severity::kWARNING> gLogWarning;
LogStream<ILogger::Severity::kINFO> gLogInfo;
LogStream<ILogger::Severity::kVERBOSE> gLogVerbose;

void TRTException::log(std::ostream& logStream) const
{
    logStream << file << " (" << line << ") - " << name << " Error in " << function << ": " << status;
    if (message != nullptr)
    {
        logStream << " (" << message << ")";
    }
    logStream << std::endl;
}

void reportValidationFailure(char const* msg, char const* file, int32_t line)
{
    gLogError << "Validation failed: " << msg << "\n" << file << ':' << line << std::endl;
}

void reportAssertion(char const* msg, char const* file, int32_t line)
{
    gLogError << "Assertion failed: " << msg << "\n" << file << ':' << line << std::endl;
    std::abort();
}

void logError(char const* msg, char const* file, char const* fn, int32_t line)
{
    gLogError << "Parameter check failed at: " << file << "::" << fn << "::" << line
              << ", condition: " << msg << std::endl;
}

[[noreturn]] void throwCudaError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    CudaError error(file, function, line, status, msg);
    error.log(gLogError);
    throw error;
}

[[noreturn]] void throwPluginError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    PluginError error(file, function, line, status, msg);
    error.log(gLogError);
    throw error;
}

[[noreturn]] void throwCublasError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    CublasError error(file, function, line, status, msg);
    error.log(gLogError);
    throw error;
}

[[noreturn]] void throwCudnnError(char const* file, char const* function, int32_t line, int32_t status, char const* msg)
{
    CudnnError error(file, function, line, status, msg);
    error.log(gLogError);
    throw error;
}

} // namespace nvinfer1::plugin
