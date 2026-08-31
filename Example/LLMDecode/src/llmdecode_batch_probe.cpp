// External-KV continuous-batching proof-of-concept.
//
// This is deliberately separate from llmdecode's production serve protocol.
// It proves the important llama.cpp primitive before introducing a request
// scheduler: independent external KV prefixes occupy independent sequence
// IDs in one context, then one llama_decode() consumes one token per active
// sequence (M = active request count).

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::string model;
    uint32_t layers = 36;
    uint32_t kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t prompt_tokens = 16;
    uint32_t warmup_steps = 4;
    uint32_t shared_steps = 8;
    uint32_t sequences = 2;
    int threads = 32;
};

struct Prefix {
    std::vector<std::vector<uint16_t>> keys;
    std::vector<std::vector<uint16_t>> values;
    std::vector<const uint16_t *> key_ptrs;
    std::vector<const uint16_t *> value_ptrs;
    std::vector<llama_pos> positions;
};

struct Sequence {
    llama_seq_id id;
    llama_token next_token;
    llama_pos next_position;
    std::vector<llama_token> tokens;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

const char * need_value(int & index, int argc, char ** argv, const char * name) {
    if (++index >= argc) fail(std::string("missing value for ") + name);
    return argv[index];
}

Options parse_args(int argc, char ** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--model") options.model = need_value(index, argc, argv, "--model");
        else if (arg == "--layers") options.layers = std::stoul(need_value(index, argc, argv, "--layers"));
        else if (arg == "--kv-heads") options.kv_heads = std::stoul(need_value(index, argc, argv, "--kv-heads"));
        else if (arg == "--head-dim") options.head_dim = std::stoul(need_value(index, argc, argv, "--head-dim"));
        else if (arg == "--prompt-tokens") options.prompt_tokens = std::stoul(need_value(index, argc, argv, "--prompt-tokens"));
        else if (arg == "--warmup-steps") options.warmup_steps = std::stoul(need_value(index, argc, argv, "--warmup-steps"));
        else if (arg == "--shared-steps") options.shared_steps = std::stoul(need_value(index, argc, argv, "--shared-steps"));
        else if (arg == "--sequences") options.sequences = std::stoul(need_value(index, argc, argv, "--sequences"));
        else if (arg == "--threads") options.threads = std::stoi(need_value(index, argc, argv, "--threads"));
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: " << argv[0] << " --model GELab.gguf [--layers 36 --kv-heads 8 --head-dim 128]\n"
                << "       [--prompt-tokens 16 --warmup-steps 4 --shared-steps 8 --sequences 2 --threads 32]\n";
            std::exit(0);
        } else {
            fail("unknown argument: " + arg);
        }
    }
    if (options.model.empty() || !options.layers || !options.kv_heads || !options.head_dim ||
            !options.prompt_tokens || !options.shared_steps || !options.sequences || options.threads < 0) {
        fail("invalid batch-probe options");
    }
    return options;
}

Prefix make_distinct_prefix(const Options & options, uint16_t half_value) {
    const size_t elements = static_cast<size_t>(options.kv_heads) * options.prompt_tokens * options.head_dim;
    Prefix prefix;
    prefix.keys.resize(options.layers);
    prefix.values.resize(options.layers);
    prefix.key_ptrs.resize(options.layers);
    prefix.value_ptrs.resize(options.layers);
    for (uint32_t layer = 0; layer < options.layers; ++layer) {
        // Different finite FP16 values make the two imported caches distinct
        // without requiring a heavyweight prefill run for this ABI probe.
        prefix.keys[layer].assign(elements, static_cast<uint16_t>(half_value + (layer & 1)));
        prefix.values[layer].assign(elements, static_cast<uint16_t>(half_value + ((layer + 1) & 1)));
        prefix.key_ptrs[layer] = prefix.keys[layer].data();
        prefix.value_ptrs[layer] = prefix.values[layer].data();
    }
    prefix.positions.assign(4 * options.prompt_tokens, 0);
    for (uint32_t token = 0; token < options.prompt_tokens; ++token) {
        prefix.positions[token] = static_cast<llama_pos>(token);
        prefix.positions[options.prompt_tokens + token] = static_cast<llama_pos>(token);
        prefix.positions[2 * options.prompt_tokens + token] = static_cast<llama_pos>(token);
    }
    return prefix;
}

void import_prefix(
    llama_context * context,
    const Options & options,
    llama_seq_id sequence_id,
    const Prefix & prefix
) {
    const llama_external_kv_f16 source = {
        options.layers,
        options.kv_heads,
        options.head_dim,
        options.prompt_tokens,
        4,
        0,
        prefix.positions.data(),
        prefix.key_ptrs.data(),
        prefix.value_ptrs.data(),
    };
    if (!llama_memory_import_kv_f16(llama_get_memory(context), sequence_id, &source)) {
        fail("llama.cpp rejected external KV import for sequence " + std::to_string(sequence_id));
    }
}

llama_token argmax(const float * logits, int32_t vocab_size) {
    if (!logits || vocab_size <= 0) fail("missing logits from llama.cpp batch decode");
    return static_cast<llama_token>(std::distance(logits, std::max_element(logits, logits + vocab_size)));
}

double decode_batch(
    llama_context * context,
    llama_batch & batch,
    std::vector<Sequence *> active,
    int32_t vocab_size
) {
    batch.n_tokens = static_cast<int32_t>(active.size());
    for (size_t index = 0; index < active.size(); ++index) {
        Sequence & sequence = *active[index];
        sequence.tokens.push_back(sequence.next_token);
        batch.token[index] = sequence.next_token;
        batch.pos[index] = sequence.next_position++;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = sequence.id;
        batch.logits[index] = 1;
    }
    const auto started = std::chrono::steady_clock::now();
    if (llama_decode(context, batch) != 0) fail("llama_decode failed for batched external-KV step");
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    for (size_t index = 0; index < active.size(); ++index) {
        active[index]->next_token = argmax(llama_get_logits_ith(context, static_cast<int32_t>(index)), vocab_size);
    }
    return seconds;
}

void print_tokens(const std::vector<llama_token> & tokens) {
    std::cout << '[';
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << tokens[index];
    }
    std::cout << ']';
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_args(argc, argv);
        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(options.model.c_str(), model_params);
        if (!model) fail("unable to load GGUF model");
        try {
            const int32_t model_layers = llama_model_n_layer(model);
            const int32_t model_kv_heads = llama_model_n_head_kv(model);
            if (model_layers != static_cast<int32_t>(options.layers) || model_kv_heads != static_cast<int32_t>(options.kv_heads)) {
                fail("GGUF geometry does not match --layers/--kv-heads");
            }
            llama_context_params context_params = llama_context_default_params();
            context_params.n_ctx = options.sequences * (options.prompt_tokens + options.warmup_steps + options.shared_steps + 4);
            context_params.n_batch = options.sequences;
            context_params.n_ubatch = options.sequences;
            context_params.n_seq_max = options.sequences;
            context_params.n_threads = options.threads;
            context_params.n_threads_batch = options.threads;
            context_params.type_k = GGML_TYPE_F16;
            context_params.type_v = GGML_TYPE_F16;
            llama_context * context = llama_init_from_model(model, context_params);
            if (!context) fail("unable to allocate shared multi-sequence llama context");
            try {
                std::vector<Prefix> prefixes;
                std::vector<Sequence> sequences;
                std::vector<Sequence *> active;
                prefixes.reserve(options.sequences);
                sequences.reserve(options.sequences);
                active.reserve(options.sequences);
                for (uint32_t index = 0; index < options.sequences; ++index) {
                    // Incrementing finite FP16 patterns prove every sequence
                    // owns a different imported cache payload.
                    prefixes.push_back(make_distinct_prefix(options, static_cast<uint16_t>(0x1400 + index * 0x0100)));
                    sequences.push_back(Sequence {
                        static_cast<llama_seq_id>(index),
                        static_cast<llama_token>(index + 1),
                        static_cast<llama_pos>(options.prompt_tokens),
                        {},
                    });
                    active.push_back(&sequences.back());
                }
                import_prefix(context, options, 0, prefixes[0]);
                llama_batch batch = llama_batch_init(static_cast<int32_t>(options.sequences), 0, static_cast<int32_t>(options.sequences));
                const int32_t vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
                double solo_seconds = 0.0;
                for (uint32_t step = 0; step < options.warmup_steps; ++step) {
                    solo_seconds += decode_batch(context, batch, {active[0]}, vocab_size);
                }
                // Remaining requests arrive after A has already decoded. Each
                // KV prefix occupies its own seq_id without touching A.
                for (uint32_t index = 1; index < options.sequences; ++index) {
                    import_prefix(context, options, static_cast<llama_seq_id>(index), prefixes[index]);
                }
                double batched_seconds = 0.0;
                for (uint32_t step = 0; step < options.shared_steps; ++step) {
                    batched_seconds += decode_batch(context, batch, active, vocab_size);
                }
                llama_batch_free(batch);
                std::cout << "{\"format\":\"llmdecode-external-kv-batch-probe-v1\""
                          << ",\"context_sequences\":" << options.sequences
                          << ",\"prompt_tokens\":" << options.prompt_tokens
                          << ",\"arrival\":\"A-import-and-decode-before-remaining-imports\""
                          << ",\"solo_steps\":" << options.warmup_steps
                          << ",\"solo_seconds\":" << solo_seconds
                          << ",\"batched_steps\":" << options.shared_steps
                          << ",\"batched_tokens\":" << options.sequences * options.shared_steps
                          << ",\"batched_seconds\":" << batched_seconds
                          << ",\"batched_tokens_per_second\":" << (static_cast<double>(options.sequences) * options.shared_steps / batched_seconds)
                          << ",\"sequence_a_tokens\":";
                print_tokens(sequences[0].tokens);
                std::cout << ",\"per_sequence_token_counts\":[";
                for (size_t index = 0; index < sequences.size(); ++index) {
                    if (index) std::cout << ',';
                    std::cout << sequences[index].tokens.size();
                }
                std::cout << ']';
                std::cout << "}" << std::endl;
            } catch (...) {
                llama_free(context);
                throw;
            }
            llama_free(context);
        } catch (...) {
            llama_model_free(model);
            throw;
        }
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "llmdecode_batch_probe: " << error.what() << std::endl;
        llama_backend_free();
        return 1;
    }
}
