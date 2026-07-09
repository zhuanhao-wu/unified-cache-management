#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include "asu_transport/asu_transport.h"
#include "trans_provider.h"

namespace UC::ASU {

class AICPUTransProvider : public TransProvider {
public:
    explicit AICPUTransProvider(const TransportConfig& config);
    ~AICPUTransProvider() override;

    Status CreateConnection(const std::string& localIp, const std::string& remoteIp, uint32_t port,
                            uint32_t qpNum, uint32_t timeout,
                            std::vector<ConnectionHandle>& connectionHandles) override;

    std::vector<Status> DeleteConnections(
        const std::vector<ConnectionHandle>& connectionHandles) override;

    std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches, uint32_t kernelCount,
                             uint32_t quietCount) override;

    Status RegisterMemory(ConnectionHandle connectionHandle,
                          const std::vector<RegisterMemoryDesc>& memoryDescs,
                          std::vector<MemHandle>& memoryHandles) override;

    std::vector<Status> UnregisterMemory(
        const std::vector<UnregisterMemoryDesc>& memoryDescs) override;

    Status AllocThread(uint32_t, const std::vector<uint32_t>&, std::vector<ThreadHandle>&) override
    {
        return Status::OK();
    }

    std::vector<Status> FreeThread(const std::vector<ThreadHandle>& threads) override
    {
        return std::vector<Status>(threads.size(), Status::OK());
    }

    Status GetMemTokenId(MemHandle memHandle, uint32_t& tokenId) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

using AICPUTransProviderSendHook =
    std::vector<Status> (*)(const std::vector<TransProvider::SendIoBatch>& ioBatches,
                            uint32_t kernelCount, uint32_t quietCount);

void SetAICPUTransProviderSendHook(AICPUTransProviderSendHook hook);
AICPUTransProviderSendHook GetAICPUTransProviderSendHook();

}  // namespace UC::ASU
