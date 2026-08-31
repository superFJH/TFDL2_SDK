#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

struct NpyArray {
    int fd = -1;
    void * mapping = MAP_FAILED;
    size_t mapping_size = 0;
    const uint8_t * data = nullptr;
    std::string descr;
    std::vector<size_t> shape;
    NpyArray() = default;
    NpyArray(const NpyArray &) = delete;
    NpyArray & operator=(const NpyArray &) = delete;
    NpyArray(NpyArray && other) noexcept { *this = std::move(other); }
    NpyArray & operator=(NpyArray && other) noexcept {
        if (this != &other) {
            close();
            fd = other.fd;
            mapping = other.mapping;
            mapping_size = other.mapping_size;
            data = other.data;
            descr = std::move(other.descr);
            shape = std::move(other.shape);
            other.fd = -1;
            other.mapping = MAP_FAILED;
            other.mapping_size = 0;
            other.data = nullptr;
        }
        return *this;
    }
    ~NpyArray() { close(); }
    void close() {
        if (mapping != MAP_FAILED) {
            munmap(mapping, mapping_size);
            mapping = MAP_FAILED;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }
    size_t element_count() const {
        return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    }
};

[[noreturn]] void fail(const std::string & message) { throw std::runtime_error(message); }

NpyArray map_npy(const std::string & path) {
    NpyArray result;
    result.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (result.fd < 0) fail("cannot open " + path + ": " + std::strerror(errno));
    struct stat st = {};
    if (fstat(result.fd, &st) != 0 || st.st_size < 10) fail("invalid NPY file: " + path);
    result.mapping_size = static_cast<size_t>(st.st_size);
    result.mapping = mmap(nullptr, result.mapping_size, PROT_READ, MAP_PRIVATE, result.fd, 0);
    if (result.mapping == MAP_FAILED) fail("cannot mmap " + path + ": " + std::strerror(errno));
    const auto * bytes = static_cast<const uint8_t *>(result.mapping);
    static constexpr uint8_t magic[] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
    if (std::memcmp(bytes, magic, sizeof(magic)) != 0) fail("not an NPY file: " + path);
    const uint8_t major = bytes[6];
    const size_t length_size = major == 1 ? 2 : (major == 2 || major == 3 ? 4 : 0);
    if (!length_size || result.mapping_size < 8 + length_size) fail("unsupported NPY version: " + path);
    size_t header_size = 0;
    for (size_t i = 0; i < length_size; ++i) header_size |= static_cast<size_t>(bytes[8 + i]) << (8 * i);
    const size_t data_offset = 8 + length_size + header_size;
    if (data_offset > result.mapping_size) fail("truncated NPY header: " + path);
    const std::string header(reinterpret_cast<const char *>(bytes + 8 + length_size), header_size);
    std::smatch match;
    if (!std::regex_search(header, match, std::regex("'descr'\\s*:\\s*'([^']+)'"))) fail("NPY dtype missing: " + path);
    result.descr = match[1].str();
    if (header.find("'fortran_order': True") != std::string::npos) fail("Fortran-ordered NPY is unsupported: " + path);
    if (!std::regex_search(header, match, std::regex("'shape'\\s*:\\s*\\(([^)]*)\\)"))) fail("NPY shape missing: " + path);
    const std::regex number_regex("[0-9]+");
    for (std::sregex_iterator it(match[1].first, match[1].second, number_regex), end; it != end; ++it) {
        result.shape.push_back(static_cast<size_t>(std::stoull(it->str())));
    }
    if (result.shape.empty()) fail("scalar NPY is unsupported: " + path);
    result.data = bytes + data_offset;
    return result;
}

size_t dtype_size(const std::string & descr) {
    if (descr == "<f2" || descr == "|f2") return 2;
    if (descr == "<f4" || descr == "|f4") return 4;
    if (descr == "<i4" || descr == "|i4") return 4;
    if (descr == "<i8" || descr == "|i8") return 8;
    return 0;
}

int argmax(const float * values, size_t size) {
    if (!values || size == 0) fail("empty logits");
    return static_cast<int>(std::distance(values, std::max_element(values, values + size)));
}

struct Options {
    std::string model;
    std::string logits;
    std::string positions;
    std::vector<std::string> keys;
    std::vector<std::string> values;
    uint32_t layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t prompt_tokens = 0;
    uint32_t max_new_tokens = 128;
    int32_t first_decode_position = -1;
    int threads = 0;
    std::string kv_cache_type = "fp16";
};

struct DecodeResult {
    double kv_import_seconds = 0.0;
    double decode_seconds = 0.0;
    uint32_t decode_calls = 0;
    std::vector<llama_token> generated;
};

void usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " --model MODEL.gguf --logits last_token_logits.npy\\n"
        << "       --layers N --kv-heads N --head-dim N --prompt-tokens N\\n"
        << "       --first-decode-position N --key layer0.npy --value layer0.npy ...\\n"
        << "       [--positions position_ids_3d.npy] [--kv-cache-type q8_0|fp16]\\n"
        << "       [--max-new-tokens N] [--threads N]\\n"
        << "  or: " << argv0 << " --serve --model MODEL.gguf [--threads N] [--kv-cache-type fp16]\\n";
}

void validate_options(const Options & o) {
    if (o.model.empty() || o.logits.empty() || !o.layers || !o.kv_heads || !o.head_dim ||
            !o.prompt_tokens || o.first_decode_position < 0 || !o.max_new_tokens ||
            o.keys.size() != o.layers || o.values.size() != o.layers) {
        fail("incomplete external KV decode arguments");
    }
    if (o.kv_cache_type != "fp16" && o.kv_cache_type != "q8_0") fail("--kv-cache-type must be fp16 or q8_0");
}

Options parse_args(int argc, char ** argv) {
    Options o;
    auto need = [&](int & index, const char * name) -> const char * {
        if (++index >= argc) fail(std::string("missing value for ") + name);
        return argv[index];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model") o.model = need(i, "--model");
        else if (arg == "--logits") o.logits = need(i, "--logits");
        else if (arg == "--positions") o.positions = need(i, "--positions");
        else if (arg == "--key") o.keys.emplace_back(need(i, "--key"));
        else if (arg == "--value") o.values.emplace_back(need(i, "--value"));
        else if (arg == "--layers") o.layers = std::stoul(need(i, "--layers"));
        else if (arg == "--kv-heads") o.kv_heads = std::stoul(need(i, "--kv-heads"));
        else if (arg == "--head-dim") o.head_dim = std::stoul(need(i, "--head-dim"));
        else if (arg == "--prompt-tokens") o.prompt_tokens = std::stoul(need(i, "--prompt-tokens"));
        else if (arg == "--first-decode-position") o.first_decode_position = std::stoi(need(i, "--first-decode-position"));
        else if (arg == "--kv-cache-type") o.kv_cache_type = need(i, "--kv-cache-type");
        else if (arg == "--max-new-tokens") o.max_new_tokens = std::stoul(need(i, "--max-new-tokens"));
        else if (arg == "--threads") o.threads = std::stoi(need(i, "--threads"));
        else if (arg == "--help" || arg == "-h") { usage(argv[0]); std::exit(0); }
        else fail("unknown argument: " + arg);
    }
    validate_options(o);
    return o;
}

void validate_kv_npy(const NpyArray & value, const Options & o, const char * kind, uint32_t layer) {
    const std::vector<size_t> expected = {1, o.kv_heads, o.prompt_tokens, o.head_dim};
    if (value.descr != "<f2" || value.shape != expected ||
            value.element_count() * dtype_size(value.descr) + static_cast<size_t>(value.data - static_cast<const uint8_t *>(value.mapping)) > value.mapping_size) {
        fail(std::string(kind) + " layer " + std::to_string(layer) + " must be C-contiguous float16 [1,Hkv,S,D]");
    }
}

std::vector<llama_pos> load_positions(const Options & options) {
    if (options.positions.empty()) return {};
    NpyArray positions = map_npy(options.positions);
    const size_t n_tokens = options.prompt_tokens;
    std::vector<llama_pos> result(4 * n_tokens, 0);
    const bool gelab_mrope_layout =
        // GELab's resident pipeline keeps MRoPE in its natural [3,S] form.
        // Its compatibility/on-disk pipeline serializes the same contiguous
        // values as [3,1,S] or [1,3,S].  All three have axis-major stride S.
        (positions.shape.size() == 2 && positions.shape[0] == 3 && positions.shape[1] >= n_tokens) ||
        (positions.shape.size() == 3 && positions.shape[0] == 3 && positions.shape[1] == 1 && positions.shape[2] >= n_tokens) ||
        (positions.shape.size() == 3 && positions.shape[0] == 1 && positions.shape[1] == 3 && positions.shape[2] >= n_tokens);
    // Older GELab resident runtimes retained only the shared/text position
    // row as [1,S] or [1,1,S].  Decode tokens occur after the prompt's text
    // tail, where the three MRoPE axes coincide, so materialize the required
    // three rows by copying that shared row.  This keeps the native ABI
    // explicit [4,S] while remaining compatible with those deployments.
    const bool gelab_scalar_position_layout =
        (positions.shape.size() == 2 && positions.shape[0] == 1 && positions.shape[1] >= n_tokens) ||
        (positions.shape.size() == 3 && positions.shape[0] == 1 && positions.shape[1] == 1 && positions.shape[2] >= n_tokens);
    if ((positions.descr == "<i8" || positions.descr == "|i8") &&
            (gelab_mrope_layout || gelab_scalar_position_layout)) {
        const auto * src = reinterpret_cast<const int64_t *>(positions.data);
        // All accepted layouts are C-contiguous with the token coordinate in
        // the final dimension.  This matters for the resident [3,S] layout:
        // indexing shape[2] there was undefined behaviour even though the
        // logical layout itself had already passed validation.
        const size_t source_stride = positions.shape.back();
        for (size_t axis = 0; axis < 3; ++axis) {
            for (size_t token = 0; token < n_tokens; ++token) {
                const int64_t value = src[(gelab_scalar_position_layout ? 0 : axis * source_stride) + token];
                if (value < std::numeric_limits<llama_pos>::min() || value > std::numeric_limits<llama_pos>::max()) fail("MRoPE position is outside llama_pos range");
                result[axis * n_tokens + token] = static_cast<llama_pos>(value);
            }
        }
        return result;
    }
    if ((positions.descr == "<i4" || positions.descr == "|i4") && positions.shape == std::vector<size_t>{4, n_tokens}) {
        std::memcpy(result.data(), positions.data, result.size() * sizeof(llama_pos));
        return result;
    }
    std::ostringstream detail;
    detail << "--positions must be int64 GELab MRoPE [3,Smax], [3,1,Smax], or [1,3,Smax] with Smax >= prompt tokens, or int32 [4,S]"
           << "; got dtype=" << positions.descr << " shape=[";
    for (size_t index = 0; index < positions.shape.size(); ++index) {
        if (index) detail << ',';
        detail << positions.shape[index];
    }
    detail << "] prompt_tokens=" << n_tokens;
    fail(detail.str());
}

std::string json_escape(const std::string & input) {
    std::string result;
    for (const char ch : input) {
        switch (ch) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result += ch; break;
        }
    }
    return result;
}

void write_result_json(std::ostream & out, const Options & options, const DecodeResult & result) {
    out << "{\"prompt_tokens\":" << options.prompt_tokens
        << ",\"first_decode_position\":" << options.first_decode_position
        << ",\"kv_cache_type\":\"" << options.kv_cache_type << "\""
        << ",\"kv_import_seconds\":" << result.kv_import_seconds
        << ",\"decode_calls\":" << result.decode_calls
        << ",\"decode_seconds\":" << result.decode_seconds
        << ",\"decode_tokens_per_second\":"
        << (result.decode_seconds > 0.0 ? static_cast<double>(result.decode_calls) / result.decode_seconds : 0.0)
        << ",\"generated_token_ids\":[";
    for (size_t i = 0; i < result.generated.size(); ++i) {
        if (i) out << ',';
        out << result.generated[i];
    }
    out << "]}";
}

DecodeResult decode_request(const Options & options, llama_model * model, bool stream_tokens) {
    std::vector<NpyArray> keys;
    std::vector<NpyArray> values;
    keys.reserve(options.layers);
    values.reserve(options.layers);
    for (uint32_t il = 0; il < options.layers; ++il) {
        keys.emplace_back(map_npy(options.keys[il]));
        values.emplace_back(map_npy(options.values[il]));
        validate_kv_npy(keys.back(), options, "key", il);
        validate_kv_npy(values.back(), options, "value", il);
    }
    NpyArray logits = map_npy(options.logits);
    if (logits.descr != "<f4" || logits.element_count() == 0 ||
            logits.element_count() * dtype_size(logits.descr) + static_cast<size_t>(logits.data - static_cast<const uint8_t *>(logits.mapping)) > logits.mapping_size) {
        fail("initial logits must be a non-empty float32 NPY array");
    }
    const std::vector<llama_pos> mrope_positions = load_positions(options);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = options.prompt_tokens + options.max_new_tokens + 1;
    cparams.n_batch = 1;
    cparams.n_ubatch = 1;
    cparams.n_seq_max = 1;
    const ggml_type kv_type = options.kv_cache_type == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    cparams.type_k = kv_type;
    cparams.type_v = kv_type;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (options.threads > 0) {
        cparams.n_threads = options.threads;
        cparams.n_threads_batch = options.threads;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) fail("unable to create llama context");
    try {
        std::vector<const uint16_t *> kptrs(options.layers);
        std::vector<const uint16_t *> vptrs(options.layers);
        for (uint32_t il = 0; il < options.layers; ++il) {
            kptrs[il] = reinterpret_cast<const uint16_t *>(keys[il].data);
            vptrs[il] = reinterpret_cast<const uint16_t *>(values[il].data);
        }
        const llama_external_kv_f16 source = {
            options.layers, options.kv_heads, options.head_dim, options.prompt_tokens,
            mrope_positions.empty() ? 1u : 4u, 0,
            mrope_positions.empty() ? nullptr : mrope_positions.data(),
            kptrs.data(), vptrs.data(),
        };
        DecodeResult result;
        const auto import_start = std::chrono::steady_clock::now();
        if (!llama_memory_import_kv_f16(llama_get_memory(ctx), 0, &source)) fail("llama.cpp rejected the external FP16 KV cache; see prior diagnostics");
        result.kv_import_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - import_start).count();
        const auto * vocab = llama_model_get_vocab(model);
        const llama_token eos = llama_vocab_eos(vocab);
        llama_token token = argmax(reinterpret_cast<const float *>(logits.data), logits.element_count());
        result.generated.reserve(options.max_new_tokens);
        llama_batch batch = llama_batch_init(1, 0, 1);
        const auto decode_start = std::chrono::steady_clock::now();
        for (uint32_t step = 0; step < options.max_new_tokens && token != eos; ++step) {
            result.generated.push_back(token);
            if (stream_tokens) std::cout << "{\"event\":\"token\",\"token_id\":" << token << "}" << std::endl;
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = options.first_decode_position + static_cast<llama_pos>(step);
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            const int status = llama_decode(ctx, batch);
            if (status != 0) fail("llama_decode failed with status " + std::to_string(status));
            ++result.decode_calls;
            token = argmax(llama_get_logits_ith(ctx, 0), llama_vocab_n_tokens(vocab));
        }
        result.decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
        llama_batch_free(batch);
        llama_free(ctx);
        return result;
    } catch (...) {
        llama_free(ctx);
        throw;
    }
}

Options read_request_descriptor(const std::string & path, const std::string & model, int threads, const std::string & cache_type) {
    std::ifstream file(path);
    if (!file) fail("cannot open worker request descriptor: " + path);
    Options o;
    o.model = model;
    o.threads = threads;
    o.kv_cache_type = cache_type;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        const size_t separator = line.find('=');
        if (separator == std::string::npos) fail("invalid worker request descriptor line");
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        if (key == "logits") o.logits = value;
        else if (key == "positions") o.positions = value;
        else if (key == "key") o.keys.push_back(value);
        else if (key == "value") o.values.push_back(value);
        else if (key == "layers") o.layers = std::stoul(value);
        else if (key == "kv_heads") o.kv_heads = std::stoul(value);
        else if (key == "head_dim") o.head_dim = std::stoul(value);
        else if (key == "prompt_tokens") o.prompt_tokens = std::stoul(value);
        else if (key == "first_decode_position") o.first_decode_position = std::stoi(value);
        else if (key == "max_new_tokens") o.max_new_tokens = std::stoul(value);
        else fail("unknown worker request descriptor key: " + key);
    }
    validate_options(o);
    return o;
}

int serve(int argc, char ** argv) {
    Options base;
    auto need = [&](int & index, const char * name) -> const char * {
        if (++index >= argc) fail(std::string("missing value for ") + name);
        return argv[index];
    };
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model") base.model = need(i, "--model");
        else if (arg == "--threads") base.threads = std::stoi(need(i, "--threads"));
        else if (arg == "--kv-cache-type") base.kv_cache_type = need(i, "--kv-cache-type");
        else if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        else fail("unknown --serve argument: " + arg);
    }
    if (base.model.empty()) fail("--serve requires --model");
    if (base.kv_cache_type != "fp16" && base.kv_cache_type != "q8_0") fail("--kv-cache-type must be fp16 or q8_0");
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(base.model.c_str(), mparams);
    if (!model) fail("unable to load GGUF model");
    std::cout << "{\"event\":\"ready\"}" << std::endl;
    std::string descriptor;
    while (std::getline(std::cin, descriptor)) {
        if (descriptor == "QUIT") break;
        if (descriptor.empty()) continue;
        try {
            const Options request = read_request_descriptor(descriptor, base.model, base.threads, base.kv_cache_type);
            const DecodeResult result = decode_request(request, model, true);
            std::cout << "{\"event\":\"result\",\"result\":";
            write_result_json(std::cout, request, result);
            std::cout << "}" << std::endl;
        } catch (const std::exception & error) {
            std::cout << "{\"event\":\"error\",\"message\":\"" << json_escape(error.what()) << "\"}" << std::endl;
        }
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--serve") return serve(argc, argv);
        const Options options = parse_args(argc, argv);
        ggml_backend_load_all();
        llama_backend_init();
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(options.model.c_str(), mparams);
        if (!model) fail("unable to load GGUF model");
        const DecodeResult result = decode_request(options, model, false);
        write_result_json(std::cout, options, result);
        std::cout << std::endl;
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llmdecode: " << error.what() << std::endl;
        return 1;
    }
}
