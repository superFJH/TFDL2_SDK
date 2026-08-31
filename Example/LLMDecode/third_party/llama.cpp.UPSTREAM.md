# llama.cpp vendor source

Source: <https://github.com/ggml-org/llama.cpp>

Pinned upstream commit: `9723942adc518b43c4b95dc4dce6906903eb5e09`
(`2026-08-30`, `hexagon: fix CPY fence bug`).

The source is vendored rather than a Git submodule so this example remains
self-contained.  It intentionally excludes upstream documentation, CI, tests,
examples and bundled vocabulary samples; the retained subset builds the
llama/ggml libraries and the HF-to-GGUF converter.  The local changes
implementing the external FP16 KV ABI are in these files:

- `third_party/llama.cpp/include/llama.h`
- `third_party/llama.cpp/src/llama-context.cpp`
- `third_party/llama.cpp/src/llama-kv-cache.h`
- `third_party/llama.cpp/src/llama-kv-cache.cpp`

When rebasing llama.cpp, retain the public `llama_memory_import_kv_f16()` API
and re-run the ARM KleidiAI build described in the parent README.
