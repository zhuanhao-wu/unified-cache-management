#include "aicpu_trans_provider.h"

#ifdef UCM_ASU_ENABLE_AICPU_PROVIDER

#if __has_include(<hcomm/hcomm_res.h>)
#include <hcomm/hcomm_res.h>
#include <hcomm/hcomm_primitives.h>
#elif __has_include(<hcomm_res.h>)
#include <hcomm_res.h>
#include <hcomm_primitives.h>
#else
#error "UCM_ASU_ENABLE_AICPU_PROVIDER requires hcomm_res.h and hcomm_primitives.h"
#endif

#include <acl/acl.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "logger.h"
#include "trans/buffer.h"

namespace UC::ASU {
namespace {

std::atomic<AICPUTransProviderSendHook> g_sendHook{nullptr};

constexpr std::uint32_t kDefaultNotifyNum = 1;
constexpr std::uint32_t kDefaultUbSqDepth = 0xFFFFFFFFU;
constexpr std::uint32_t kDefaultSendTimeoutMs = 1836U * 1000U;
constexpr std::uint32_t kAclSyncGraceMs = 5000U;
constexpr std::uint32_t kCpuKernelMode = 0U;
constexpr std::uint32_t kKernelBlockDim = 1U;
constexpr std::uintptr_t kHostRegisterAlignment = 4096U;
constexpr const char* kDefaultChannelName = "ucm_asu_aicpu";
constexpr const char* kProviderSignature =
    "UCM_ASU_AICPU_STAGED_URMA_HOST_MAPPED_PAYLOAD_MR_V4";
constexpr const char* kBatchSendKernelName = "HixlBatchSend";
constexpr const char* kDefaultAscendHome = "/usr/local/Ascend/cann";
constexpr const char* kHixlKernelJsonSuffix =
    "/opp/built-in/op_impl/aicpu/config/libcann_hixl_kernel.json";

struct UcmHixlSendIoBatch {
    ::ChannelHandle channel;
    const void* local_src;
    std::uint64_t len;
};

struct UcmHixlBatchSendParam {
    ::ThreadHandle thread;
    UcmHixlSendIoBatch* io_batches;
    std::uint64_t batch_size;
    std::uint32_t* status_array;
    std::uint32_t timeout_ms;
    void* stats;
    std::uint32_t complete_sender_cqe;
};

template <typename T>
auto SetHcommChannelNameIfSupported(T& desc, const char* channelName, int)
    -> decltype((void)(desc.channelName = channelName), void())
{
    desc.channelName = channelName;
}

template <typename T>
void SetHcommChannelNameIfSupported(T&, const char*, ...)
{
}

struct ConnectionRecord {
    ::ChannelHandle channel{0};
    ::ThreadHandle thread{0};
    HcommA5UdmaSendChannelContext sendContext{};
    bool sendContextUsable{false};
    void* deviceSendContext{nullptr};
    bool staged{false};
    bool hasImmOverride{false};
    std::uint32_t immOverride{0};
    HcommStagedChannelInfo stagedInfo{};
};

struct MemoryRecord {
    HcommMemHandle mem{nullptr};
    EndpointHandle endpoint{nullptr};
    std::string tag;
    std::uintptr_t originalAddr{0};
    std::uintptr_t localAddr{0};
    std::uintptr_t transportAddr{0};
    std::size_t size{0};
    TransProvider::MemType memoryType{TransProvider::MemType::MEM_HOST};
    bool ownsHostMapping{false};
    std::uintptr_t hostMappingBase{0};
    std::uint32_t tokenId{0};
    bool hasToken{false};
    std::uint32_t stagedMrId{0};
    bool stagedPublished{false};
};

Status HcommError(const std::string& op, HcommResult ret, StatusCode code = StatusCode::INTERNAL_ERROR)
{
    auto message = op + " failed ret=" + std::to_string(ret);
    UC_ERROR("AICPUTransProvider: {}", message);
    return Status::Error(code, std::move(message));
}

Status HcommConnectionError(const std::string& op, HcommResult ret)
{
    return HcommError(op, ret, StatusCode::CONNECTION_ERROR);
}

Status AclError(const std::string& op, aclError ret, StatusCode code = StatusCode::INTERNAL_ERROR)
{
    std::string message = op + " failed ret=" + std::to_string(static_cast<int>(ret));
    if (const char* recent = aclGetRecentErrMsg(); recent != nullptr && recent[0] != '\0') {
        message += " msg=";
        message += recent;
    }
    UC_ERROR("AICPUTransProvider: {}", message);
    return Status::Error(code, std::move(message));
}

std::string Normalize(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string GetAttr(const std::unordered_map<std::string, std::string>& attrs,
                    std::initializer_list<const char*> names)
{
    for (const auto* name : names) {
        auto it = attrs.find(name);
        if (it != attrs.end()) { return it->second; }
    }
    return {};
}

std::string GetConfigAttr(const TransportConfig& config, std::initializer_list<const char*> names)
{
    return GetAttr(config.attrs, names);
}

std::string GetEndpointAttr(const AsuEndpoint* endpoint, std::initializer_list<const char*> names)
{
    return endpoint == nullptr ? std::string{} : GetAttr(endpoint->attrs, names);
}

std::uint32_t ParseUint32(std::string value, std::uint32_t fallback)
{
    if (value.empty()) { return fallback; }
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 0);
    if (errno != 0 || end == value.c_str() || *end != '\0' ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
        return fallback;
    }
    return static_cast<std::uint32_t>(parsed);
}

std::uint64_t ParseUint64(std::string value, std::uint64_t fallback)
{
    if (value.empty()) { return fallback; }
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 0);
    if (errno != 0 || end == value.c_str() || *end != '\0') { return fallback; }
    return static_cast<std::uint64_t>(parsed);
}

std::uint16_t ParseUint16(std::string value, std::uint16_t fallback)
{
    const auto parsed = ParseUint32(std::move(value), fallback);
    if (parsed > std::numeric_limits<std::uint16_t>::max()) { return fallback; }
    return static_cast<std::uint16_t>(parsed);
}

bool ParseBool(std::string value, bool fallback)
{
    if (value.empty()) { return fallback; }
    value = Normalize(std::move(value));
    if (value == "1" || value == "true" || value == "yes" || value == "on") { return true; }
    if (value == "0" || value == "false" || value == "no" || value == "off") { return false; }
    return fallback;
}

const char* CommProtocolName(CommProtocol protocol)
{
    switch (protocol) {
        case COMM_PROTOCOL_ROCE: return "roce";
        case COMM_PROTOCOL_UBC_TP: return "ubc_tp";
        case COMM_PROTOCOL_UB_MEM: return "ub_mem";
        case COMM_PROTOCOL_UBOE: return "uboe";
        case COMM_PROTOCOL_UBC_CTP: return "ubc_ctp";
        default: return "unknown";
    }
}

bool IsUbProtocol(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_UBC_TP || protocol == COMM_PROTOCOL_UB_MEM ||
           protocol == COMM_PROTOCOL_UBOE || protocol == COMM_PROTOCOL_UBC_CTP;
}

const char* CommAddrTypeName(CommAddrType type)
{
    switch (type) {
        case COMM_ADDR_TYPE_IP_V4: return "ipv4";
        case COMM_ADDR_TYPE_IP_V6: return "ipv6";
        case COMM_ADDR_TYPE_ID: return "id";
        case COMM_ADDR_TYPE_EID: return "eid";
        default: return "unknown";
    }
}

int HexNibble(char value)
{
    if (value >= '0' && value <= '9') { return value - '0'; }
    if (value >= 'a' && value <= 'f') { return value - 'a' + 10; }
    if (value >= 'A' && value <= 'F') { return value - 'A' + 10; }
    return -1;
}

std::string NormalizeEidLiteral(const std::string& text)
{
    std::string input = text;
    if (input.size() > 4U && Normalize(input.substr(0, 4U)) == "eid:") {
        input = input.substr(4U);
    }
    std::string normalized;
    normalized.reserve(input.size());
    std::size_t start = 0U;
    if (input.size() > 2U && input[0] == '0' && (input[1] == 'x' || input[1] == 'X')) {
        start = 2U;
    }
    for (std::size_t index = start; index < input.size(); ++index) {
        if (input[index] != ':' && input[index] != '-') {
            normalized.push_back(input[index]);
        }
    }
    return normalized;
}

bool TryFillEidCommAddr(const std::string& text, CommAddr& out)
{
    const std::string normalized = NormalizeEidLiteral(text);
    if (normalized.size() != COMM_ADDR_EID_LEN * 2U) { return false; }

    std::uint8_t eid[COMM_ADDR_EID_LEN]{};
    for (std::size_t index = 0U; index < COMM_ADDR_EID_LEN; ++index) {
        const int high = HexNibble(normalized[index * 2U]);
        const int low = HexNibble(normalized[index * 2U + 1U]);
        if (high < 0 || low < 0) { return false; }
        eid[index] = static_cast<std::uint8_t>((high << 4U) | low);
    }

    out.type = COMM_ADDR_TYPE_EID;
    std::memset(out.raws, 0, sizeof(out.raws));
    std::memcpy(out.eid, eid, sizeof(eid));
    return true;
}

std::uint32_t ResolveDeviceId(const TransportConfig& config)
{
    const auto explicitDevice = GetConfigAttr(config, {"device_id", "deviceId", "logical_device_id"});
    if (!explicitDevice.empty()) { return ParseUint32(explicitDevice, 0); }
    for (const auto& endpoint : config.endpoints) {
        if (endpoint.deviceId >= 0) { return static_cast<std::uint32_t>(endpoint.deviceId); }
    }
    const char* env = std::getenv("UMC_ASU_DEVICE_ID");
    return env == nullptr ? 0 : ParseUint32(env, 0);
}

std::uint32_t ResolveRemoteDeviceId(const AsuEndpoint* endpoint, std::uint32_t fallback)
{
    const auto explicitDevice = GetEndpointAttr(endpoint, {"device_id", "deviceId", "remote_device_id"});
    if (!explicitDevice.empty()) { return ParseUint32(explicitDevice, fallback); }
    if (endpoint != nullptr && endpoint->deviceId >= 0) {
        return static_cast<std::uint32_t>(endpoint->deviceId);
    }
    return fallback;
}

CommProtocol ResolveProtocol(const TransportConfig& config, const AsuEndpoint* endpoint)
{
    auto value = GetEndpointAttr(endpoint, {"hcomm_protocol", "aicpu_hcomm_protocol", "protocol"});
    if (value.empty()) {
        value = GetConfigAttr(config, {"hcomm_protocol", "aicpu_hcomm_protocol", "protocol"});
    }
    value = Normalize(std::move(value));
    if (value == "roce") { return COMM_PROTOCOL_ROCE; }
    if (value == "ubc_tp" || value == "ub_tp") { return COMM_PROTOCOL_UBC_TP; }
    if (value == "ub_mem" || value == "ubmem") { return COMM_PROTOCOL_UB_MEM; }
    if (value == "uboe") { return COMM_PROTOCOL_UBOE; }
    return COMM_PROTOCOL_UBC_CTP;
}

bool ResolveStagedEnabled(const TransportConfig& config, const AsuEndpoint* endpoint)
{
    auto value = GetEndpointAttr(endpoint, {"aicpu_staged_oob", "staged_oob", "hcomm_staged_oob"});
    if (value.empty()) {
        value = GetConfigAttr(config, {"aicpu_staged_oob", "staged_oob", "hcomm_staged_oob"});
    }
    return ParseBool(value, false);
}

std::string ResolveStagedOobHost(const TransportConfig& config, const AsuEndpoint* endpoint,
                                 const std::string& remoteIp)
{
    auto host = GetEndpointAttr(endpoint, {"aicpu_staged_oob_host", "staged_oob_host", "oob_host"});
    if (host.empty()) {
        host = GetConfigAttr(config, {"aicpu_staged_oob_host", "staged_oob_host", "oob_host"});
    }
    return host.empty() ? remoteIp : host;
}

std::uint16_t ResolveStagedOobPort(const TransportConfig& config, const AsuEndpoint* endpoint,
                                   std::uint32_t port)
{
    auto value = GetEndpointAttr(endpoint, {"aicpu_staged_oob_port", "staged_oob_port", "oob_port"});
    if (value.empty()) {
        value = GetConfigAttr(config, {"aicpu_staged_oob_port", "staged_oob_port", "oob_port"});
    }
    return ParseUint16(value, static_cast<std::uint16_t>(port));
}

std::uint32_t ResolveStagedClientId(const TransportConfig& config, const AsuEndpoint* endpoint)
{
    auto value = GetEndpointAttr(endpoint, {"aicpu_staged_client_id", "staged_client_id", "client_id"});
    if (value.empty()) {
        value = GetConfigAttr(config, {"aicpu_staged_client_id", "staged_client_id", "client_id"});
    }
    return ParseUint32(value, static_cast<std::uint32_t>(config.asuId));
}

EndpointLocType ResolveLocType(const TransportConfig& config, const AsuEndpoint* endpoint)
{
    auto value = GetEndpointAttr(endpoint, {"endpoint_loc", "placement", "loc"});
    if (value.empty()) { value = GetConfigAttr(config, {"endpoint_loc", "placement", "loc"}); }
    value = Normalize(std::move(value));
    return value == "host" ? ENDPOINT_LOC_TYPE_HOST : ENDPOINT_LOC_TYPE_DEVICE;
}

Status FillCommAddr(const std::string& text, CommAddr& out)
{
    if (TryFillEidCommAddr(text, out)) {
        UC_INFO("AICPUTransProvider: parsed HCOMM endpoint address as EID addr={}", text);
        return Status::OK();
    }
    if (inet_pton(AF_INET, text.c_str(), &out.addr) == 1) {
        out.type = COMM_ADDR_TYPE_IP_V4;
        return Status::OK();
    }
    if (inet_pton(AF_INET6, text.c_str(), &out.addr6) == 1) {
        out.type = COMM_ADDR_TYPE_IP_V6;
        return Status::OK();
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(text.c_str(), &end, 0);
    if (errno == 0 && end != text.c_str() && *end == '\0' &&
        parsed <= std::numeric_limits<std::uint32_t>::max()) {
        out.type = COMM_ADDR_TYPE_ID;
        out.id = static_cast<std::uint32_t>(parsed);
        return Status::OK();
    }
    return Status::Error(StatusCode::INVALID_ARGUMENT, "invalid hcomm endpoint address: " + text);
}

Status BuildEndpointDesc(const TransportConfig& config, const AsuEndpoint* endpoint,
                         const std::string& addr, std::uint32_t deviceId, CommProtocol protocol,
                         EndpointDesc& out)
{
    const auto init = EndpointDescInit(&out, 1U);
    if (init != 0) { return HcommError("EndpointDescInit", init); }

    out.protocol = protocol;
    auto status = FillCommAddr(addr, out.commAddr);
    if (!status.ok()) { return status; }
    UC_DEBUG("AICPUTransProvider: EndpointDesc resolved addr={} addr_type={} device_id={} "
             "protocol={}",
             addr, CommAddrTypeName(out.commAddr.type), deviceId, CommProtocolName(protocol));

    out.loc.locType = ResolveLocType(config, endpoint);
    if (out.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
        out.loc.device.devPhyId = deviceId;
        out.loc.device.superDevId =
            ParseUint32(GetEndpointAttr(endpoint, {"super_device_id", "superDevId"}), 0);
        out.loc.device.serverIdx =
            ParseUint32(GetEndpointAttr(endpoint, {"server_idx", "serverIdx"}), 0);
        out.loc.device.superPodIdx =
            ParseUint32(GetEndpointAttr(endpoint, {"super_pod_idx", "superPodIdx"}), 0);
    } else {
        out.loc.host.id = ParseUint32(GetEndpointAttr(endpoint, {"host_id", "hostId"}), 0);
    }
    return Status::OK();
}

CommMemType ToHcommMemType(TransProvider::MemType type)
{
    return type == TransProvider::MemType::MEM_HOST ? COMM_MEM_TYPE_HOST : COMM_MEM_TYPE_DEVICE;
}

std::string ResolveHixlKernelJsonPath(const TransportConfig& config)
{
    auto path = GetConfigAttr(config, {"hixl_kernel_json", "aicpu_hixl_kernel_json"});
    if (!path.empty()) { return path; }
    if (const char* env = std::getenv("HIXL_KERNEL_JSON"); env != nullptr && env[0] != '\0') {
        return env;
    }
    const char* ascendHome = std::getenv("ASCEND_HOME_PATH");
    path = (ascendHome == nullptr || ascendHome[0] == '\0') ? kDefaultAscendHome : ascendHome;
    path += kHixlKernelJsonSuffix;
    return path;
}

int32_t MakeAclSyncTimeoutMs(std::uint32_t hcommTimeoutMs)
{
    const std::uint32_t effective =
        hcommTimeoutMs == 0U ? kDefaultSendTimeoutMs : hcommTimeoutMs;
    const std::uint64_t timeout = static_cast<std::uint64_t>(effective) + kAclSyncGraceMs;
    const auto max = static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max());
    return static_cast<int32_t>(timeout > max ? max : timeout);
}

std::uint32_t MakeKernelTimeoutSeconds(std::uint32_t hcommTimeoutMs)
{
    return static_cast<std::uint32_t>(MakeAclSyncTimeoutMs(hcommTimeoutMs) / 1000) + 1U;
}

struct DeviceAllocation {
    void* ptr{nullptr};

    DeviceAllocation() = default;
    explicit DeviceAllocation(void* value) : ptr(value) {}
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
    ~DeviceAllocation()
    {
        if (ptr != nullptr) { (void)aclrtFree(ptr); }
    }
};

ConnectionRecord* ToConnectionRecord(TransProvider::ConnectionHandle handle)
{
    return static_cast<ConnectionRecord*>(handle);
}

MemoryRecord* ToMemoryRecord(TransProvider::MemHandle handle)
{
    return static_cast<MemoryRecord*>(handle);
}

}  // namespace

struct AICPUTransProvider::Impl {
    struct HostMapping {
        std::size_t size{0};
        std::uintptr_t deviceAddr{0};
        std::size_t refCount{0};
    };

    explicit Impl(const TransportConfig& configIn)
        : config(configIn),
          localDeviceId(ResolveDeviceId(configIn)),
          notifyNum(ParseUint32(GetConfigAttr(configIn, {"aicpu_notify_num", "notify_num"}),
                                kDefaultNotifyNum)),
          ubSqDepth(ParseUint32(GetConfigAttr(configIn, {"aicpu_ub_sq_depth", "ub_sq_depth"}),
                                kDefaultUbSqDepth)),
          qos(ParseUint32(GetConfigAttr(configIn, {"aicpu_qos", "qos"}), 0)),
          sendTimeoutMs(ParseUint32(GetConfigAttr(configIn, {"aicpu_send_timeout_ms", "send_timeout_ms",
                                                            "timeout"}),
                                    kDefaultSendTimeoutMs)),
          immData(ParseUint32(GetConfigAttr(configIn, {"aicpu_send_imm", "aicpu_imm_data", "imm_data"}),
                              0)),
          sendWithImm(ParseBool(GetConfigAttr(configIn, {"aicpu_send_with_imm", "send_with_imm"}), true)),
          completeSenderCqe(ParseBool(GetConfigAttr(configIn, {"aicpu_complete_sender_cqe",
                                                               "complete_sender_cqe"}),
                                      true)),
          channelName(GetConfigAttr(configIn, {"aicpu_channel_name", "channel_name"})),
          hixlKernelJsonPath(ResolveHixlKernelJsonPath(configIn)),
          stagedEnabled(ResolveStagedEnabled(configIn, nullptr)),
          stagedPublishMems(ParseBool(GetConfigAttr(configIn, {"aicpu_staged_publish_mrs",
                                                               "staged_publish_mrs"}),
                                      true)),
          stagedChannelUseSeg1(ParseBool(GetConfigAttr(configIn, {"aicpu_staged_channel_seg1",
                                                                  "staged_channel_seg1"}),
                                         false)),
          stagedKato(ParseUint32(GetConfigAttr(configIn, {"aicpu_staged_kato", "staged_kato"}), 0)),
          stagedRmUasid(ParseUint32(GetConfigAttr(configIn, {"aicpu_staged_rm_uasid",
                                                             "staged_rm_uasid"}),
                                    0)),
          stagedMamiTag(ParseUint64(GetConfigAttr(configIn, {"aicpu_staged_mami_tag",
                                                             "staged_mami_tag"}),
                                    0))
    {
        if (channelName.empty()) { channelName = kDefaultChannelName; }
    }

    ~Impl()
    {
        if (hixlBin != nullptr || stream != nullptr || endpoint != nullptr) {
            const auto deviceStatus = EnsureAclDeviceBound("AICPUTransProvider cleanup");
            if (!deviceStatus.ok()) {
                UC_WARN("AICPUTransProvider: cleanup continuing after device bind failure: {}",
                        deviceStatus.message);
            }
        }
        if (hixlBin != nullptr) {
            (void)aclrtBinaryUnLoad(hixlBin);
            hixlBin = nullptr;
        }
        if (stream != nullptr) {
            (void)aclrtDestroyStream(stream);
            stream = nullptr;
        }
        if (endpoint != nullptr) {
            (void)HcommEndpointDestroy(endpoint);
            endpoint = nullptr;
        }
    }

    const AsuEndpoint* FindEndpoint(const std::string& remoteIp, std::uint32_t port) const
    {
        auto byAddr = std::find_if(config.endpoints.begin(), config.endpoints.end(),
                                   [&](const AsuEndpoint& endpoint) {
                                       return endpoint.ip == remoteIp && endpoint.port == port;
                                   });
        if (byAddr != config.endpoints.end()) { return &*byAddr; }

        byAddr = std::find_if(config.endpoints.begin(), config.endpoints.end(),
                              [&](const AsuEndpoint& endpoint) {
                                  return endpoint.ip == remoteIp;
                              });
        return byAddr == config.endpoints.end() ? nullptr : &*byAddr;
    }

    Status EnsureEndpointLocked(const std::string& localIp, CommProtocol protocol)
    {
        if (endpoint != nullptr) {
            if (endpointProtocol != protocol) {
                UC_ERROR("AICPUTransProvider: mixed hcomm protocols are not supported "
                         "existing_protocol={} requested_protocol={} endpoint_ip={}",
                         CommProtocolName(endpointProtocol), CommProtocolName(protocol), endpointIp);
                return Status::Error(StatusCode::INVALID_ARGUMENT,
                                     "AICPUTransProvider: mixed hcomm protocols are not supported");
            }
            UC_DEBUG("AICPUTransProvider: reusing HCOMM endpoint local_addr={} protocol={}",
                     endpointIp, CommProtocolName(protocol));
            return Status::OK();
        }

        auto resolvedLocalIp = localIp;
        if (resolvedLocalIp.empty()) {
            resolvedLocalIp = GetConfigAttr(config, {"localIp", "local_ip", "aicpu_local_ip"});
        }
        if (resolvedLocalIp.empty()) {
            UC_ERROR("AICPUTransProvider: localIp is required for hcomm endpoint");
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "AICPUTransProvider: localIp is required for hcomm endpoint");
        }

        UC_INFO("AICPUTransProvider: creating HCOMM endpoint local_addr={} local_device_id={} "
                "protocol={}",
                resolvedLocalIp, localDeviceId, CommProtocolName(protocol));
        EndpointDesc localDesc{};
        auto status = BuildEndpointDesc(config, nullptr, resolvedLocalIp, localDeviceId, protocol, localDesc);
        if (!status.ok()) {
            UC_ERROR("AICPUTransProvider: local EndpointDesc build failed local_addr={} "
                     "local_device_id={} protocol={} message={}",
                     resolvedLocalIp, localDeviceId, CommProtocolName(protocol), status.message);
            return status;
        }

        EndpointHandle created = nullptr;
        const auto ret = HcommEndpointCreate(&localDesc, &created);
        if (ret != 0) { return HcommConnectionError("HcommEndpointCreate", ret); }

        endpoint = created;
        endpointIp = resolvedLocalIp;
        endpointProtocol = protocol;
        UC_INFO("AICPUTransProvider: HCOMM endpoint created local_addr={} local_device_id={} "
                "protocol={}",
                endpointIp, localDeviceId, CommProtocolName(endpointProtocol));
        return Status::OK();
    }

    // HCOMM resolves UB runtime resources through the current thread's ACL device.
    Status EnsureAclDeviceBound(const char* stage)
    {
        if (localDeviceId > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "AICPUTransProvider: invalid local device id " +
                                     std::to_string(localDeviceId));
        }

        const auto deviceId = static_cast<std::int32_t>(localDeviceId);
        thread_local std::int32_t readyDeviceId = -1;

        std::int32_t currentDevice = -1;
        if (readyDeviceId == deviceId) {
            const auto getRet = aclrtGetDevice(&currentDevice);
            if (getRet == ACL_SUCCESS && currentDevice == deviceId) { return Status::OK(); }
        }

        const auto initRet = aclInit(nullptr);
        if (initRet != ACL_SUCCESS && initRet != ACL_ERROR_REPEAT_INITIALIZE) {
            return AclError(std::string("aclInit before ") + stage, initRet);
        }

        currentDevice = -1;
        const auto getRet = aclrtGetDevice(&currentDevice);
        if (getRet == ACL_SUCCESS && currentDevice == deviceId) {
            readyDeviceId = deviceId;
            return Status::OK();
        }

        const auto setRet = aclrtSetDevice(deviceId);
        if (setRet != ACL_SUCCESS) {
            return AclError(std::string("aclrtSetDevice before ") + stage +
                                " device_id=" + std::to_string(deviceId),
                            setRet);
        }

        UC_INFO("AICPUTransProvider: aclrtSetDevice bound device_id={} stage={} "
                "previous_device={} aclrtGetDevice_ret={} aclInit_ret={}",
                deviceId, stage, currentDevice, static_cast<int>(getRet),
                static_cast<int>(initRet));
        readyDeviceId = deviceId;
        return Status::OK();
    }

    Status PublishMemoryStaged(MemoryRecord& record)
    {
        bool enabled = false;
        bool publishMems = false;
        std::string oobHost;
        std::uint16_t oobPort = 0;
        std::vector<std::uint32_t> clientIds;
        {
            std::lock_guard<std::mutex> lock(mu);
            enabled = stagedEnabled;
            publishMems = stagedPublishMems;
            oobHost = activeStagedOobHost;
            oobPort = activeStagedOobPort;
            for (auto* conn : connections) {
                if (conn != nullptr && conn->staged && conn->stagedInfo.clientId != 0U) {
                    clientIds.push_back(conn->stagedInfo.clientId);
                }
            }
            if (clientIds.empty() && activeStagedClientId != 0U) {
                clientIds.push_back(activeStagedClientId);
            }
        }
        if (!enabled || !publishMems || record.stagedPublished) {
            return Status::OK();
        }
        std::sort(clientIds.begin(), clientIds.end());
        clientIds.erase(std::unique(clientIds.begin(), clientIds.end()), clientIds.end());
        if (oobHost.empty() || oobPort == 0U || clientIds.empty()) {
            UC_ERROR("AICPUTransProvider: staged MR publish requires an active staged connection "
                     "oob_host={} oob_port={} client_ids={}",
                     oobHost, oobPort, clientIds.size());
            return Status::Error(StatusCode::CONNECTION_ERROR,
                                 "AICPUTransProvider: staged MR publish requires an active staged connection");
        }

        HcommStagedMrDesc mr{};
        mr.memHandle = record.mem;
        mr.mrId = record.stagedMrId;

        for (std::size_t index = 0; index < clientIds.size(); ++index) {
            const auto clientId = clientIds[index];
            const auto requestId = record.stagedMrId + 1U + static_cast<std::uint32_t>(index);
            UC_INFO("AICPUTransProvider: publishing staged MR tag={} mr_id={} original_addr={} "
                    "transport_addr={} size={} oob={}:{} client_id={} request_id={}",
                    record.tag, record.stagedMrId, record.originalAddr, record.transportAddr,
                    record.size, oobHost, oobPort, clientId, requestId);

            HcommStagedMrPublishDesc publish{};
            const auto initRet = HcommStagedMrPublishDescInit(&publish, 1U);
            if (initRet != 0) { return HcommError("HcommStagedMrPublishDescInit", initRet); }
            publish.oobHost = oobHost.c_str();
            publish.oobPort = oobPort;
            publish.timeoutMs = sendTimeoutMs;
            publish.clientId = clientId;
            publish.requestId = requestId;
            publish.mrNum = 1U;
            publish.mrs = &mr;
            publish.useSeg1Handshake = 1U;

            const auto ret = HcommMemPublishStaged(record.endpoint, &publish);
            if (ret != 0) { return HcommConnectionError("HcommMemPublishStaged", ret); }
        }
        record.stagedPublished = true;
        UC_INFO("AICPUTransProvider: staged MR published tag={} mr_id={} original_addr={} "
                "transport_addr={} size={} token_id={} has_token={} client_count={}",
                record.tag, record.stagedMrId, record.originalAddr, record.transportAddr,
                record.size, record.tokenId, record.hasToken ? 1 : 0, clientIds.size());
        return Status::OK();
    }

    Status AcquireHostMapping(std::uintptr_t hostAddr, std::size_t size,
                              std::uintptr_t& mappingBase, std::uintptr_t& deviceAddr)
    {
        mappingBase = hostAddr & ~(kHostRegisterAlignment - 1U);
        const auto offset = hostAddr - mappingBase;
        if (size > std::numeric_limits<std::size_t>::max() - offset) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "AICPUTransProvider: aligned host mapping size overflows");
        }
        const auto mappingSize = size + static_cast<std::size_t>(offset);

        std::lock_guard<std::mutex> lock(hostMappingMu);
        auto iter = hostMappings.find(mappingBase);
        if (iter != hostMappings.end()) {
            if (iter->second.size < mappingSize) {
                return Status::Error(
                    StatusCode::BUFFER_NOT_SUPPORTED,
                    "AICPUTransProvider: host mapping overlaps a smaller active mapping");
            }
            ++iter->second.refCount;
            deviceAddr = iter->second.deviceAddr + offset;
            UC_INFO("AICPUTransProvider: reused host mapping host_addr={} mapping_base={} "
                    "device_addr={} mapping_size={} request_size={} ref_count={}",
                    hostAddr, mappingBase, deviceAddr, iter->second.size, size,
                    iter->second.refCount);
            return Status::OK();
        }

        void* mapped = nullptr;
        const auto mapStatus = UC::Trans::Buffer::RegisterHostBuffer(
            reinterpret_cast<void*>(mappingBase), mappingSize, &mapped);
        if (mapStatus.Failure() || mapped == nullptr) {
            return Status::Error(StatusCode::BUFFER_NOT_SUPPORTED,
                                 "AICPUTransProvider: host mapping failed status=" +
                                     mapStatus.ToString());
        }

        const auto mappedBase = reinterpret_cast<std::uintptr_t>(mapped);
        deviceAddr = mappedBase + offset;
        hostMappings.emplace(mappingBase, HostMapping{mappingSize, mappedBase, 1});
        UC_INFO("AICPUTransProvider: created host mapping host_addr={} mapping_base={} "
                "mapped_base={} device_addr={} mapping_size={} request_size={} device_id={}",
                hostAddr, mappingBase, mappedBase, deviceAddr, mappingSize, size, localDeviceId);
        return Status::OK();
    }

    void ReleaseHostMapping(std::uintptr_t mappingBase)
    {
        std::lock_guard<std::mutex> lock(hostMappingMu);
        auto iter = hostMappings.find(mappingBase);
        if (iter == hostMappings.end()) {
            UC_WARN("AICPUTransProvider: host mapping release missed mapping_base={}",
                    mappingBase);
            return;
        }
        if (--iter->second.refCount != 0) {
            UC_INFO("AICPUTransProvider: retained host mapping mapping_base={} mapped_base={} "
                    "ref_count={}",
                    mappingBase, iter->second.deviceAddr, iter->second.refCount);
            return;
        }
        UC::Trans::Buffer::UnregisterHostBuffer(reinterpret_cast<void*>(mappingBase));
        UC_INFO("AICPUTransProvider: released host mapping mapping_base={} mapped_base={} "
                "mapping_size={}",
                mappingBase, iter->second.deviceAddr, iter->second.size);
        hostMappings.erase(iter);
    }

    Status ReleaseMemoryRecord(MemoryRecord& record)
    {
        Status status = Status::OK();
        if (record.mem != nullptr) {
            const auto ret = HcommMemUnreg(record.endpoint, record.mem);
            record.mem = nullptr;
            if (ret != 0) {
                status = HcommError("HcommMemUnreg", ret, StatusCode::BUFFER_NOT_REGISTERED);
            }
        }
        if (record.ownsHostMapping) {
            ReleaseHostMapping(record.hostMappingBase);
            record.ownsHostMapping = false;
        }
        return status;
    }

    std::uint64_t SendContextFlags() const
    {
        std::uint64_t flags = completeSenderCqe ? HCOMM_A5_UDMA_SEND_CONTEXT_FLAG_SENDER_CQE : 0U;
        if (sendWithImm) { flags |= HCOMM_A5_UDMA_SEND_CONTEXT_FLAG_WITH_IMM; }
        return flags;
    }

    Status EnsureDeviceSendContextLocked(ConnectionRecord& record)
    {
        const auto effectiveImm = record.hasImmOverride ? record.immOverride : immData;
        UC_DEBUG("AICPUTransProvider: initializing A5 send context channel={} effective_imm={} "
                 "has_imm_override={} send_with_imm={} complete_sender_cqe={}",
                 record.channel, effectiveImm, record.hasImmOverride ? 1 : 0,
                 sendWithImm ? 1 : 0, completeSenderCqe ? 1 : 0);
        const auto initRet =
            HcommA5UdmaSendContextInit(record.channel, effectiveImm, SendContextFlags(), &record.sendContext);
        if (initRet != 0) { return HcommConnectionError("HcommA5UdmaSendContextInit", initRet); }

        if (record.deviceSendContext == nullptr) {
            void* deviceContext = nullptr;
            const auto mallocRet =
                aclrtMalloc(&deviceContext, sizeof(record.sendContext), ACL_MEM_MALLOC_NORMAL_ONLY);
            if (mallocRet != ACL_SUCCESS) { return AclError("aclrtMalloc A5 send context", mallocRet); }
            record.deviceSendContext = deviceContext;
        }

        const auto copyRet =
            aclrtMemcpy(record.deviceSendContext, sizeof(record.sendContext), &record.sendContext,
                        sizeof(record.sendContext), ACL_MEMCPY_HOST_TO_DEVICE);
        if (copyRet != ACL_SUCCESS) { return AclError("aclrtMemcpy A5 send context", copyRet); }
        record.sendContextUsable = true;
        UC_DEBUG("AICPUTransProvider: A5 send context ready channel={} device_context={}",
                 record.channel, record.deviceSendContext);
        return Status::OK();
    }

    Status EnsureAclStreamLocked()
    {
        if (stream != nullptr) { return Status::OK(); }
        const auto ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS) { return AclError("aclrtCreateStream", ret); }
        return Status::OK();
    }

    Status LoadHixlBatchSendLocked(aclrtFuncHandle& func)
    {
        if (hixlBin == nullptr) {
            aclrtBinaryLoadOption option{};
            option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
            option.value.cpuKernelMode = kCpuKernelMode;
            aclrtBinaryLoadOptions options{};
            options.options = &option;
            options.numOpt = 1U;
            const auto ret =
                aclrtBinaryLoadFromFile(hixlKernelJsonPath.c_str(), &options, &hixlBin);
            if (ret != ACL_SUCCESS) {
                return AclError("aclrtBinaryLoadFromFile " + hixlKernelJsonPath, ret);
            }
        }

        const auto getRet = aclrtBinaryGetFunction(hixlBin, kBatchSendKernelName, &func);
        if (getRet != ACL_SUCCESS) { return AclError("aclrtBinaryGetFunction HixlBatchSend", getRet); }
        return Status::OK();
    }

    Status LaunchBatchSendLocked(const std::vector<TransProvider::SendIoBatch>& ioBatches,
                                 std::vector<std::uint32_t>& hixlStatuses)
    {
        hixlStatuses.assign(ioBatches.size(), 1U);
        auto status = EnsureAclStreamLocked();
        if (!status.ok()) { return status; }

        std::vector<UcmHixlSendIoBatch> batches;
        batches.reserve(ioBatches.size());
        ::ThreadHandle thread = 0;
        for (const auto& io : ioBatches) {
            auto* conn = ToConnectionRecord(io.connectionHandle);
            if (conn == nullptr || !conn->sendContextUsable || conn->deviceSendContext == nullptr) {
                return Status::Error(StatusCode::CONNECTION_ERROR,
                                     "AICPUTransProvider::Send: A5 send context is not ready");
            }
            if (thread == 0) {
                thread = conn->thread;
            } else if (thread != conn->thread) {
                return Status::Error(StatusCode::UNSUPPORTED,
                                     "AICPUTransProvider::Send: mixed AICPU threads in one batch "
                                     "are not supported by HixlBatchSend");
            }
            batches.push_back(UcmHixlSendIoBatch{
                reinterpret_cast<::ChannelHandle>(conn->deviceSendContext), io.sendBuffer, io.len});
        }
        if (thread == 0) {
            return Status::Error(StatusCode::CONNECTION_ERROR,
                                 "AICPUTransProvider::Send: no AICPU thread available");
        }

        void* deviceBatchesRaw = nullptr;
        auto ret = aclrtMalloc(&deviceBatchesRaw, batches.size() * sizeof(UcmHixlSendIoBatch),
                               ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) { return AclError("aclrtMalloc HIXL batch entries", ret); }
        DeviceAllocation deviceBatches(deviceBatchesRaw);

        void* deviceStatusRaw = nullptr;
        ret = aclrtMalloc(&deviceStatusRaw, hixlStatuses.size() * sizeof(std::uint32_t),
                          ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) { return AclError("aclrtMalloc HIXL status array", ret); }
        DeviceAllocation deviceStatus(deviceStatusRaw);

        ret = aclrtMemcpy(deviceBatches.ptr, batches.size() * sizeof(UcmHixlSendIoBatch),
                          batches.data(), batches.size() * sizeof(UcmHixlSendIoBatch),
                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) { return AclError("aclrtMemcpy HIXL batch entries", ret); }

        aclrtFuncHandle func = nullptr;
        status = LoadHixlBatchSendLocked(func);
        if (!status.ok()) { return status; }

        UcmHixlBatchSendParam param{};
        param.thread = thread;
        param.io_batches = static_cast<UcmHixlSendIoBatch*>(deviceBatches.ptr);
        param.batch_size = static_cast<std::uint64_t>(batches.size());
        param.status_array = static_cast<std::uint32_t*>(deviceStatus.ptr);
        param.timeout_ms = sendTimeoutMs;
        param.stats = nullptr;
        param.complete_sender_cqe = completeSenderCqe ? 1U : 0U;

        aclrtArgsHandle args = nullptr;
        aclrtParamHandle paramHandle = nullptr;
        ret = aclrtKernelArgsInit(func, &args);
        if (ret != ACL_SUCCESS) { return AclError("aclrtKernelArgsInit HixlBatchSend", ret); }
        ret = aclrtKernelArgsAppend(args, &param, sizeof(param), &paramHandle);
        if (ret != ACL_SUCCESS) { return AclError("aclrtKernelArgsAppend HixlBatchSend", ret); }
        ret = aclrtKernelArgsFinalize(args);
        if (ret != ACL_SUCCESS) { return AclError("aclrtKernelArgsFinalize HixlBatchSend", ret); }

        aclrtLaunchKernelAttr attr{};
        attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
        attr.value.timeout = MakeKernelTimeoutSeconds(sendTimeoutMs);
        aclrtLaunchKernelCfg cfg{};
        cfg.numAttrs = 1U;
        cfg.attrs = &attr;
        ret = aclrtLaunchKernelWithConfig(func, kKernelBlockDim, stream, &cfg, args, nullptr);
        if (ret != ACL_SUCCESS) { return AclError("aclrtLaunchKernelWithConfig HixlBatchSend", ret); }
        ret = aclrtSynchronizeStreamWithTimeout(stream, MakeAclSyncTimeoutMs(sendTimeoutMs));
        if (ret != ACL_SUCCESS) {
            return AclError("aclrtSynchronizeStreamWithTimeout HixlBatchSend", ret);
        }

        ret = aclrtMemcpy(hixlStatuses.data(), hixlStatuses.size() * sizeof(std::uint32_t),
                          deviceStatus.ptr, hixlStatuses.size() * sizeof(std::uint32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) { return AclError("aclrtMemcpy HIXL status array", ret); }
        return Status::OK();
    }

    TransportConfig config;
    std::uint32_t localDeviceId{0};
    std::uint32_t notifyNum{kDefaultNotifyNum};
    std::uint32_t ubSqDepth{kDefaultUbSqDepth};
    std::uint32_t qos{0};
    std::uint32_t sendTimeoutMs{kDefaultSendTimeoutMs};
    std::uint32_t immData{0};
    bool sendWithImm{true};
    bool completeSenderCqe{true};
    std::string channelName;
    std::string hixlKernelJsonPath;
    bool stagedEnabled{false};
    bool stagedPublishMems{true};
    bool stagedChannelUseSeg1{false};
    std::uint32_t stagedKato{0};
    std::uint32_t stagedRmUasid{0};
    std::uint64_t stagedMamiTag{0};
    std::string activeStagedOobHost;
    std::uint16_t activeStagedOobPort{0};
    std::uint32_t activeStagedClientId{0};
    std::uint32_t nextStagedMrId{0};

    std::mutex mu;
    EndpointHandle endpoint{nullptr};
    std::string endpointIp;
    CommProtocol endpointProtocol{COMM_PROTOCOL_RESERVED};
    std::unordered_set<ConnectionRecord*> connections;
    std::unordered_set<MemoryRecord*> memories;
    std::uint64_t nextMemTag{1};

    std::mutex hostMappingMu;
    std::unordered_map<std::uintptr_t, HostMapping> hostMappings;

    std::mutex aclMu;
    aclrtStream stream{nullptr};
    aclrtBinHandle hixlBin{nullptr};
};

void SetAICPUTransProviderSendHook(AICPUTransProviderSendHook hook)
{
    g_sendHook.store(hook, std::memory_order_release);
}

AICPUTransProviderSendHook GetAICPUTransProviderSendHook()
{
    return g_sendHook.load(std::memory_order_acquire);
}

AICPUTransProvider::AICPUTransProvider(const TransportConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
    UC_INFO("AICPU_TRANSPORT_PROVIDER_SIGNATURE={} asu_id={} local_device_id={} "
            "staged_enabled={} staged_publish_mrs={} channel_name={}",
            kProviderSignature, impl_->config.asuId, impl_->localDeviceId,
            impl_->stagedEnabled ? 1 : 0, impl_->stagedPublishMems ? 1 : 0,
            impl_->channelName);
}

AICPUTransProvider::~AICPUTransProvider()
{
    if (!impl_) { return; }

    std::vector<UnregisterMemoryDesc> memDescs;
    std::vector<ConnectionHandle> connHandles;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        memDescs.reserve(impl_->memories.size());
        for (auto* mem : impl_->memories) { memDescs.push_back({nullptr, mem}); }
        connHandles.reserve(impl_->connections.size());
        for (auto* conn : impl_->connections) { connHandles.push_back(conn); }
    }

    if (!memDescs.empty()) { (void)UnregisterMemory(memDescs); }
    if (!connHandles.empty()) { (void)DeleteConnections(connHandles); }
}

Status AICPUTransProvider::CreateConnection(const std::string& localIp, const std::string& remoteIp,
                                            uint32_t port, uint32_t qpNum, uint32_t timeout,
                                            std::vector<ConnectionHandle>& connectionHandles)
{
    connectionHandles.clear();
    if (qpNum == 0) { return Status::OK(); }

    auto status = impl_->EnsureAclDeviceBound("CreateConnection");
    if (!status.ok()) { return status; }

    const auto* endpoint = impl_->FindEndpoint(remoteIp, port);
    const auto protocol = ResolveProtocol(impl_->config, endpoint);
    const auto remoteDeviceId = ResolveRemoteDeviceId(endpoint, impl_->localDeviceId);
    UC_INFO("AICPUTransProvider: CreateConnection start signature={} local_addr={} remote_addr={} "
            "remote_port={} qp_num={} timeout_ms={} local_device_id={} remote_device_id={} "
            "protocol={} endpoint_matched={}",
            kProviderSignature, localIp, remoteIp, port, qpNum, timeout, impl_->localDeviceId,
            remoteDeviceId, CommProtocolName(protocol), endpoint == nullptr ? 0 : 1);

    EndpointDesc remoteDesc{};
    status = BuildEndpointDesc(impl_->config, endpoint, remoteIp, remoteDeviceId, protocol, remoteDesc);
    if (!status.ok()) {
        UC_ERROR("AICPUTransProvider: remote EndpointDesc build failed remote_addr={} "
                 "remote_port={} remote_device_id={} protocol={} message={}",
                 remoteIp, port, remoteDeviceId, CommProtocolName(protocol), status.message);
        return status;
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        status = impl_->EnsureEndpointLocked(localIp, protocol);
        if (!status.ok()) {
            UC_ERROR("AICPUTransProvider: EnsureEndpointLocked failed local_addr={} "
                     "local_device_id={} protocol={} message={}",
                     localIp, impl_->localDeviceId, CommProtocolName(protocol), status.message);
            return status;
        }
    }

    std::vector<ConnectionHandle> createdHandles;
    createdHandles.reserve(qpNum);
    const bool staged = ResolveStagedEnabled(impl_->config, endpoint) || impl_->stagedEnabled;
    const std::string stagedOobHost = ResolveStagedOobHost(impl_->config, endpoint, remoteIp);
    const std::uint16_t stagedOobPort = ResolveStagedOobPort(impl_->config, endpoint, port);
    const std::uint32_t stagedClientId = ResolveStagedClientId(impl_->config, endpoint);
    UC_INFO("AICPUTransProvider: CreateConnection resolved staged={} staged_oob={}:{} "
            "staged_client_id={} notify_num={} ub_sq_depth={} qos={} send_with_imm={} "
            "complete_sender_cqe={}",
            staged ? 1 : 0, stagedOobHost, stagedOobPort, stagedClientId, impl_->notifyNum,
            impl_->ubSqDepth, impl_->qos, impl_->sendWithImm ? 1 : 0,
            impl_->completeSenderCqe ? 1 : 0);

    for (std::uint32_t remaining = qpNum; remaining > 0; --remaining) {
        const std::uint32_t qpIndex = static_cast<std::uint32_t>(createdHandles.size());
        UC_DEBUG("AICPUTransProvider: creating connection handle qp_index={} qp_num={} staged={}",
                 qpIndex, qpNum, staged ? 1 : 0);
        auto* record = new ConnectionRecord{};
        const std::uint32_t notify = impl_->notifyNum;
        const auto threadRet = HcommThreadAlloc(COMM_ENGINE_AICPU, 1U, &notify, &record->thread);
        if (threadRet != 0) {
            delete record;
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommConnectionError("HcommThreadAlloc", threadRet);
        }

        HcommChannelDesc desc{};
        const auto descRet = HcommChannelDescInit(&desc, 1U);
        if (descRet != 0) {
            (void)HcommThreadFree(&record->thread, 1U);
            delete record;
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommError("HcommChannelDescInit", descRet);
        }
        desc.remoteEndpoint = remoteDesc;
        desc.notifyNum = notify;
        desc.exchangeAllMems = true;
        desc.role = HCOMM_SOCKET_ROLE_CLIENT;
        desc.port = ParseUint16(GetEndpointAttr(endpoint, {"aicpu_port", "hcomm_port"}),
                                static_cast<std::uint16_t>(port));
        desc.ubAttr.sqDepth = impl_->ubSqDepth;
        desc.qos = impl_->qos;
        SetHcommChannelNameIfSupported(desc, impl_->channelName.c_str(), 0);

        EndpointHandle hcommEndpoint = nullptr;
        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            hcommEndpoint = impl_->endpoint;
        }
        HcommResult chanRet = 0;
        if (staged) {
            HcommStagedChannelDesc stagedDesc{};
            const auto stagedInitRet = HcommStagedChannelDescInit(&stagedDesc, 1U);
            if (stagedInitRet != 0) {
                (void)HcommThreadFree(&record->thread, 1U);
                delete record;
                if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
                return HcommError("HcommStagedChannelDescInit", stagedInitRet);
            }
            stagedDesc.channelDesc = desc;
            stagedDesc.oobHost = stagedOobHost.c_str();
            stagedDesc.oobPort = stagedOobPort;
            stagedDesc.timeoutMs = impl_->sendTimeoutMs;
            stagedDesc.clientId = stagedClientId;
            stagedDesc.qpIndex = qpIndex;
            stagedDesc.kato = impl_->stagedKato;
            stagedDesc.useSeg1Handshake = impl_->stagedChannelUseSeg1 ? 1U : 0U;
            stagedDesc.connMode = HCOMM_STAGED_CONN_MODE_RM;
            stagedDesc.rmUasid = impl_->stagedRmUasid;
            stagedDesc.mamiTag = impl_->stagedMamiTag;
            UC_INFO("AICPUTransProvider: HcommChannelCreateStaged begin qp_index={} "
                    "oob={}:{} client_id={} timeout_ms={} use_seg1={} rm_uasid={} "
                    "mami_tag={} channel_port={}",
                    qpIndex, stagedOobHost, stagedOobPort, stagedClientId, stagedDesc.timeoutMs,
                    stagedDesc.useSeg1Handshake, stagedDesc.rmUasid, stagedDesc.mamiTag,
                    desc.port);
            chanRet = HcommChannelCreateStaged(hcommEndpoint, COMM_ENGINE_AICPU, &stagedDesc, 1U,
                                               &record->channel);
        } else {
            UC_INFO("AICPUTransProvider: HcommChannelCreate begin qp_index={} channel_port={}",
                    qpIndex, desc.port);
            chanRet = HcommChannelCreate(hcommEndpoint, COMM_ENGINE_AICPU, &desc, 1U, &record->channel);
        }
        if (chanRet != 0) {
            (void)HcommThreadFree(&record->thread, 1U);
            delete record;
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return HcommConnectionError(staged ? "HcommChannelCreateStaged" : "HcommChannelCreate",
                                        chanRet);
        }
        UC_INFO("AICPUTransProvider: channel created qp_index={} staged={} channel={} thread={}",
                qpIndex, staged ? 1 : 0, record->channel, record->thread);

        if (staged) {
            HcommStagedChannelInfo stagedInfo{};
            const auto infoInitRet = HcommStagedChannelInfoInit(&stagedInfo, 1U);
            if (infoInitRet != 0) {
                auto channel = record->channel;
                (void)HcommChannelDestroy(&channel, 1U);
                (void)HcommThreadFree(&record->thread, 1U);
                delete record;
                if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
                return HcommError("HcommStagedChannelInfoInit", infoInitRet);
            }
            const auto infoRet = HcommChannelGetStagedInfo(record->channel, &stagedInfo);
            if (infoRet != 0) {
                auto channel = record->channel;
                (void)HcommChannelDestroy(&channel, 1U);
                (void)HcommThreadFree(&record->thread, 1U);
                delete record;
                if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
                return HcommConnectionError("HcommChannelGetStagedInfo", infoRet);
            }
            record->staged = true;
            record->stagedInfo = stagedInfo;
            record->hasImmOverride = true;
            record->immOverride = stagedInfo.sendImm;
            UC_INFO("AICPUTransProvider: staged channel info qp_index={} controller_id={} "
                    "send_imm={} client_id={} configured_client_id={} remote_jetty_id={} "
                    "remote_token_value={}",
                    stagedInfo.qpIndex, stagedInfo.controllerId, stagedInfo.sendImm,
                    stagedInfo.clientId, stagedClientId, stagedInfo.remoteJettyId,
                    stagedInfo.remoteTokenValue);
            {
                std::lock_guard<std::mutex> lock(impl_->mu);
                impl_->stagedEnabled = true;
                impl_->activeStagedOobHost = stagedOobHost;
                impl_->activeStagedOobPort = stagedOobPort;
                impl_->activeStagedClientId =
                    stagedInfo.clientId == 0U ? stagedClientId : stagedInfo.clientId;
            }
        }

        {
            std::lock_guard<std::mutex> lock(impl_->aclMu);
            status = impl_->EnsureDeviceSendContextLocked(*record);
        }
        if (!status.ok()) {
            if (record->deviceSendContext != nullptr) {
                (void)aclrtFree(record->deviceSendContext);
                record->deviceSendContext = nullptr;
            }
            auto channel = record->channel;
            (void)HcommChannelDestroy(&channel, 1U);
            (void)HcommThreadFree(&record->thread, 1U);
            delete record;
            if (!createdHandles.empty()) { (void)DeleteConnections(createdHandles); }
            return status;
        }

        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            impl_->connections.insert(record);
        }
        createdHandles.push_back(record);
        connectionHandles.push_back(record);
        UC_INFO("AICPUTransProvider: connection handle ready qp_index={} created_handles={}/{}",
                qpIndex, connectionHandles.size(), qpNum);
    }

    UC_INFO("AICPUTransProvider: CreateConnection complete remote_addr={} remote_port={} "
            "handles={}",
            remoteIp, port, connectionHandles.size());
    return Status::OK();
}

std::vector<Status> AICPUTransProvider::DeleteConnections(
    const std::vector<ConnectionHandle>& connectionHandles)
{
    std::vector<Status> results(connectionHandles.size(), Status::OK());
    if (!connectionHandles.empty()) {
        const auto deviceStatus = impl_->EnsureAclDeviceBound("DeleteConnections");
        if (!deviceStatus.ok()) {
            return std::vector<Status>(connectionHandles.size(), deviceStatus);
        }
    }
    for (std::size_t index = 0; index < connectionHandles.size(); ++index) {
        auto* record = ToConnectionRecord(connectionHandles[index]);
        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            if (record == nullptr || impl_->connections.find(record) == impl_->connections.end()) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "AICPUTransProvider: invalid connection handle");
                continue;
            }
            impl_->connections.erase(record);
        }

        Status status = Status::OK();
        if (record->deviceSendContext != nullptr) {
            std::lock_guard<std::mutex> lock(impl_->aclMu);
            const auto ret = aclrtFree(record->deviceSendContext);
            record->deviceSendContext = nullptr;
            if (ret != ACL_SUCCESS) { status = AclError("aclrtFree A5 send context", ret); }
        }
        if (record->channel != 0) {
            auto channel = record->channel;
            const auto ret = HcommChannelDestroy(&channel, 1U);
            if (ret != 0) { status = HcommConnectionError("HcommChannelDestroy", ret); }
        }
        if (record->thread != 0) {
            auto thread = record->thread;
            const auto ret = HcommThreadFree(&thread, 1U);
            if (ret != 0 && status.ok()) { status = HcommConnectionError("HcommThreadFree", ret); }
        }
        results[index] = status;
        delete record;
    }
    return results;
}

std::vector<Status> AICPUTransProvider::Send(const std::vector<SendIoBatch>& ioBatches,
                                             uint32_t kernelCount, uint32_t quietCount)
{
    (void)kernelCount;
    (void)quietCount;
    if (ioBatches.empty()) { return {}; }

    std::vector<Status> results(ioBatches.size(), Status::OK());
    bool valid = true;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        for (std::size_t index = 0; index < ioBatches.size(); ++index) {
            const auto& item = ioBatches[index];
            auto* conn = ToConnectionRecord(item.connectionHandle);
            if (conn == nullptr || impl_->connections.find(conn) == impl_->connections.end()) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "AICPUTransProvider::Send: invalid connection handle");
                valid = false;
                continue;
            }
            if (item.sendBuffer == nullptr || item.len == 0) {
                results[index] = Status::Error(StatusCode::INVALID_ARGUMENT,
                                               "AICPUTransProvider::Send: empty send buffer");
                valid = false;
            }
        }
    }
    if (!valid) { return results; }

    if (auto hook = GetAICPUTransProviderSendHook()) {
        return hook(ioBatches, kernelCount, quietCount);
    }

    const auto deviceStatus = impl_->EnsureAclDeviceBound("Send");
    if (!deviceStatus.ok()) { return std::vector<Status>(ioBatches.size(), deviceStatus); }

    std::vector<std::uint32_t> hixlStatuses;
    Status launchStatus = Status::OK();
    {
        std::lock_guard<std::mutex> lock(impl_->aclMu);
        launchStatus = impl_->LaunchBatchSendLocked(ioBatches, hixlStatuses);
    }
    if (!launchStatus.ok()) { return std::vector<Status>(ioBatches.size(), launchStatus); }

    for (std::size_t index = 0; index < hixlStatuses.size(); ++index) {
        if (hixlStatuses[index] == 0U) { continue; }
        results[index] = Status::Error(StatusCode::INTERNAL_ERROR,
                                       "HixlBatchSend failed for batch index " +
                                           std::to_string(index) + " status=" +
                                           std::to_string(hixlStatuses[index]));
    }
    return results;
}

Status AICPUTransProvider::RegisterMemory(ConnectionHandle,
                                          const std::vector<RegisterMemoryDesc>& memoryDescs,
                                          std::vector<MemHandle>& memoryHandles)
{
    memoryHandles.clear();
    if (memoryDescs.empty()) { return Status::OK(); }

    auto bindStatus = impl_->EnsureAclDeviceBound("RegisterMemory");
    if (!bindStatus.ok()) { return bindStatus; }

    EndpointHandle endpoint = nullptr;
    CommProtocol endpointProtocol = COMM_PROTOCOL_RESERVED;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        endpoint = impl_->endpoint;
        endpointProtocol = impl_->endpointProtocol;
    }
    if (endpoint == nullptr) {
        UC_ERROR("AICPUTransProvider: RegisterMemory requires an established HCOMM endpoint "
                 "desc_count={}",
                 memoryDescs.size());
        return Status::Error(StatusCode::CONNECTION_ERROR,
                             "AICPUTransProvider::RegisterMemory requires an established hcomm endpoint");
    }

    UC_INFO("AICPUTransProvider: RegisterMemory begin desc_count={}", memoryDescs.size());
    std::vector<MemoryRecord*> created;
    created.reserve(memoryDescs.size());
    auto cleanupCreated = [&]() {
        std::vector<UnregisterMemoryDesc> cleanup;
        cleanup.reserve(created.size());
        for (auto* existing : created) { cleanup.push_back({nullptr, existing}); }
        if (!cleanup.empty()) { (void)UnregisterMemory(cleanup); }
        created.clear();
        memoryHandles.clear();
    };
    for (const auto& desc : memoryDescs) {
        if (desc.addr == 0 || desc.size == 0) {
            UC_ERROR("AICPUTransProvider: RegisterMemory invalid desc addr={} size={} type={}",
                     desc.addr, desc.size, static_cast<int>(desc.memoryType));
            cleanupCreated();
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "AICPUTransProvider::RegisterMemory: zero addr/size in desc");
        }

        auto* record = new MemoryRecord{};
        record->endpoint = endpoint;
        record->originalAddr = desc.addr;
        record->localAddr = desc.localAddr == 0 ? desc.addr : desc.localAddr;
        record->transportAddr = desc.addr;
        record->size = desc.size;
        record->memoryType = desc.memoryType;
        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            record->tag = impl_->channelName + ":mem:" + std::to_string(impl_->nextMemTag++);
            record->stagedMrId = impl_->nextStagedMrId++;
        }

        if (desc.memoryType == MemType::MEM_HOST && IsUbProtocol(endpointProtocol)) {
            auto mappingStatus = impl_->AcquireHostMapping(
                desc.addr, desc.size, record->hostMappingBase, record->transportAddr);
            if (!mappingStatus.ok()) {
                delete record;
                cleanupCreated();
                return mappingStatus;
            }
            record->ownsHostMapping = true;
        }

        CommMem mem{};
        mem.type = record->ownsHostMapping ? COMM_MEM_TYPE_DEVICE : ToHcommMemType(desc.memoryType);
        mem.addr = reinterpret_cast<void*>(record->transportAddr);
        mem.size = static_cast<std::uint64_t>(desc.size);
        UC_INFO("AICPUTransProvider: HcommMemReg begin tag={} original_addr={} local_addr={} "
                "transport_addr={} size={} input_mem_type={} hcomm_mem_type={} protocol={} "
                "owns_host_mapping={} staged_mr_id={}",
                record->tag, record->originalAddr, record->localAddr, record->transportAddr,
                record->size, static_cast<int>(desc.memoryType), static_cast<int>(mem.type),
                CommProtocolName(endpointProtocol), record->ownsHostMapping ? 1 : 0,
                record->stagedMrId);
        const auto ret = HcommMemReg(endpoint, record->tag.c_str(), &mem, &record->mem);
        if (ret != 0) {
            (void)impl_->ReleaseMemoryRecord(*record);
            delete record;
            cleanupCreated();
            return HcommError("HcommMemReg", ret, StatusCode::BUFFER_NOT_SUPPORTED);
        }

        HcommMemTokenInfo tokenInfo{};
        const auto tokenRet = HcommMemGetTokenInfo(record->mem, &tokenInfo);
        if (tokenRet == 0) {
            if (tokenInfo.addr != record->transportAddr || tokenInfo.size != record->size) {
                UC_ERROR("AICPUTransProvider: HCOMM token range mismatch tag={} "
                         "transport_addr={} transport_size={} token_addr={} token_size={}",
                         record->tag, record->transportAddr, record->size, tokenInfo.addr,
                         tokenInfo.size);
                (void)impl_->ReleaseMemoryRecord(*record);
                delete record;
                cleanupCreated();
                return Status::Error(StatusCode::BUFFER_NOT_SUPPORTED,
                                     "AICPUTransProvider: HCOMM token range mismatch");
            }
            if (tokenInfo.type == HCOMM_MEM_TOKEN_TYPE_UB && tokenInfo.tokenValue != 0U) {
                record->tokenId = tokenInfo.tokenValue;
                record->hasToken = true;
            } else if (tokenInfo.type == HCOMM_MEM_TOKEN_TYPE_RDMA && tokenInfo.rkey != 0U) {
                record->tokenId = tokenInfo.rkey;
                record->hasToken = true;
            }
            UC_INFO("AICPUTransProvider: HcommMemGetTokenInfo tag={} token_type={} token_id={} "
                    "token_addr={} token_size={} has_token={}",
                    record->tag, static_cast<int>(tokenInfo.type), record->tokenId,
                    tokenInfo.addr, tokenInfo.size, record->hasToken ? 1 : 0);
        } else {
            UC_WARN("AICPUTransProvider: HcommMemGetTokenInfo failed tag={} ret={}",
                    record->tag, tokenRet);
        }

        std::vector<ConnectionRecord*> connections;
        bool hasStagedConnection = false;
        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            connections.assign(impl_->connections.begin(), impl_->connections.end());
            for (auto* conn : connections) {
                if (conn != nullptr && conn->staged) {
                    hasStagedConnection = true;
                    break;
                }
            }
        }
        for (auto* conn : connections) {
            if (conn == nullptr || conn->channel == 0) { continue; }
            HcommMemHandle memHandle = record->mem;
            if (conn->staged) {
                UC_INFO("AICPUTransProvider: HcommChannelUpdateStagedLocalMemInfo begin "
                        "tag={} staged_mr_id={} channel={}",
                        record->tag, record->stagedMrId, conn->channel);
                const auto updateRet =
                    HcommChannelUpdateStagedLocalMemInfo(&memHandle, 1U, conn->channel);
                if (updateRet != 0) {
                    (void)impl_->ReleaseMemoryRecord(*record);
                    delete record;
                    cleanupCreated();
                    return HcommConnectionError("HcommChannelUpdateStagedLocalMemInfo", updateRet);
                }
            } else if (!hasStagedConnection) {
                UC_DEBUG("AICPUTransProvider: HcommChannelUpdateMemInfo begin tag={} channel={}",
                         record->tag, conn->channel);
                const auto updateRet = HcommChannelUpdateMemInfo(&memHandle, 1U, conn->channel);
                if (updateRet != 0) {
                    (void)impl_->ReleaseMemoryRecord(*record);
                    delete record;
                    cleanupCreated();
                    return HcommConnectionError("HcommChannelUpdateMemInfo", updateRet);
                }
            }
        }

        auto publishStatus = impl_->PublishMemoryStaged(*record);
        if (!publishStatus.ok()) {
            (void)impl_->ReleaseMemoryRecord(*record);
            delete record;
            cleanupCreated();
            return publishStatus;
        }

        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            impl_->memories.insert(record);
        }
        created.push_back(record);
        memoryHandles.push_back(record);
        UC_INFO("AICPUTransProvider: RegisterMemory record ready tag={} original_addr={} "
                "transport_addr={} token_id={} has_token={} memory_handles={}/{}",
                record->tag, record->originalAddr, record->transportAddr, record->tokenId,
                record->hasToken ? 1 : 0, memoryHandles.size(), memoryDescs.size());
    }

    UC_INFO("AICPUTransProvider: RegisterMemory complete handles={}", memoryHandles.size());
    return Status::OK();
}

std::vector<Status> AICPUTransProvider::UnregisterMemory(
    const std::vector<UnregisterMemoryDesc>& memoryDescs)
{
    std::vector<Status> results(memoryDescs.size(), Status::OK());
    if (!memoryDescs.empty()) {
        const auto deviceStatus = impl_->EnsureAclDeviceBound("UnregisterMemory");
        if (!deviceStatus.ok()) { return std::vector<Status>(memoryDescs.size(), deviceStatus); }
    }
    for (std::size_t index = 0; index < memoryDescs.size(); ++index) {
        auto* record = ToMemoryRecord(memoryDescs[index].memoryHandle);
        {
            std::lock_guard<std::mutex> lock(impl_->mu);
            if (record == nullptr || impl_->memories.find(record) == impl_->memories.end()) {
                results[index] = Status::Error(StatusCode::BUFFER_NOT_REGISTERED,
                                               "AICPUTransProvider: invalid memory handle");
                continue;
            }
            impl_->memories.erase(record);
        }

        results[index] = impl_->ReleaseMemoryRecord(*record);
        delete record;
    }
    return results;
}

Status AICPUTransProvider::GetMemTokenId(MemHandle memHandle, uint32_t& tokenId)
{
    tokenId = 0;
    auto* record = ToMemoryRecord(memHandle);
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (record == nullptr || impl_->memories.find(record) == impl_->memories.end()) {
        return Status::Error(StatusCode::BUFFER_NOT_REGISTERED,
                             "AICPUTransProvider::GetMemTokenId: memory handle not found");
    }

    if (!record->hasToken) {
        return Status::Error(StatusCode::UNSUPPORTED,
                             "AICPUTransProvider::GetMemTokenId: registered hcomm memory did "
                             "not expose a UB token value or RDMA rkey");
    }
    tokenId = record->tokenId;
    return Status::OK();
}

Status AICPUTransProvider::GetMemTransportAddr(MemHandle memHandle,
                                               std::uintptr_t& transportAddr) const
{
    transportAddr = 0;
    auto* record = ToMemoryRecord(memHandle);
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (record == nullptr || impl_->memories.find(record) == impl_->memories.end()) {
        return Status::Error(
            StatusCode::BUFFER_NOT_REGISTERED,
            "AICPUTransProvider::GetMemTransportAddr: memory handle not found");
    }
    transportAddr = record->transportAddr;
    return Status::OK();
}

}  // namespace UC::ASU

#endif  // UCM_ASU_ENABLE_AICPU_PROVIDER
