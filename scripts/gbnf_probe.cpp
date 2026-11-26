// Grammar probe for the denial-free compilation: parse a GBNF file with
// llama.cpp's own parser and, optionally, match text against it.
//
// Build (no model, no generation; llama.cpp internals are exported by libllama):
//   g++ -std=c++17 -O1 -I<llama.cpp>/src -I<llama.cpp>/ggml/include \
//       -I<llama.cpp>/include scripts/gbnf_probe.cpp \
//       -L<llama.cpp>/build/bin -lllama -Wl,-rpath,<llama.cpp>/build/bin -o gbnf_probe
//
// Usage:
//   gbnf_probe --parse GRAMMAR.gbnf
//   gbnf_probe --match GRAMMAR.gbnf CASES.bin
//
// CASES.bin is a sequence of records: one byte 'A' (must accept) or 'R' (must
// reject), then a little-endian uint32 byte length, then the UTF-8 text.
#include "llama-grammar.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static bool read_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "rb");
    if (!f) { perror(path); return false; }
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    return true;
}

static bool match_string(llama_grammar * grammar, const std::string & text) {
    try {
        llama_grammar_accept_token(*grammar, 0, text);
    } catch (const std::runtime_error &) {
        return false;
    }
    auto & stacks = llama_grammar_get_stacks(grammar);
    if (stacks.empty()) return false;
    for (const auto & stack : stacks) {
        if (stack.empty()) return true;  // the grammar can terminate here
    }
    return false;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 3) {
        fprintf(stderr, "usage: %s --parse GRAMMAR | --match GRAMMAR CASES\n", argv[0]);
        return 2;
    }
    std::string grammar_text;
    if (!read_file(argv[2], grammar_text)) return 2;
    if (strcmp(argv[1], "--parse") == 0) {
        llama_grammar_parser parser;
        const bool ok = parser.parse(grammar_text.c_str());
        printf("%s rules=%zu\n", ok ? "PARSE_OK" : "PARSE_FAIL", parser.rules.size());
        return ok ? 0 : 1;
    }
    if (strcmp(argv[1], "--match") != 0 || argc != 4) {
        fprintf(stderr, "unknown mode %s\n", argv[1]);
        return 2;
    }
    llama_grammar * grammar = llama_grammar_init_impl(nullptr, grammar_text.c_str(), "root",
                                                      false, nullptr, 0, nullptr, 0);
    if (!grammar) { fprintf(stderr, "GRAMMAR_BUILD_FAIL\n"); return 1; }
    const llama_grammar_stacks initial = llama_grammar_get_stacks(grammar);
    std::string cases;
    if (!read_file(argv[3], cases)) return 2;

    size_t offset = 0, accepted = 0, rejected = 0, mismatch = 0;
    while (offset + 5 <= cases.size()) {
        const char expect = cases[offset];
        uint32_t length = 0;
        memcpy(&length, cases.data() + offset + 1, 4);
        offset += 5;
        if (offset + length > cases.size()) { fprintf(stderr, "truncated case\n"); return 2; }
        const std::string text = cases.substr(offset, length);
        offset += length;
        llama_grammar_get_stacks(grammar) = initial;
        const bool got = match_string(grammar, text);
        const bool want = expect == 'A';
        if (got) ++accepted; else ++rejected;
        if (got != want) {
            ++mismatch;
            if (mismatch <= 20) {
                printf("MISMATCH expected=%c got=%c text=%s\n", expect, got ? 'A' : 'R', text.c_str());
            }
        }
    }
    llama_grammar_free_impl(grammar);
    printf("cases=%zu accepted=%zu rejected=%zu mismatches=%zu\n",
           accepted + rejected, accepted, rejected, mismatch);
    return mismatch == 0 ? 0 : 1;
}
