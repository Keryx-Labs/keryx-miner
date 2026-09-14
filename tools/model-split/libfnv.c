#include <stdint.h>
#include <stddef.h>
uint64_t fnv1a64(const uint8_t *data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) { h ^= data[i]; h *= 0x100000001b3ULL; }
    return h;
}
