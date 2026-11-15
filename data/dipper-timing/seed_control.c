
#define _GNU_SOURCE
#include <time.h>
#include <stdlib.h>
#include <dlfcn.h>
time_t time(time_t *out) {
    const char *value = getenv("DIPPER_SHUFFLE_SEED");
    time_t result;
    if (value) result = (time_t)strtoll(value,0,10);
    else { time_t (*original)(time_t*) = dlsym(RTLD_NEXT,"time"); result=original(0); }
    if(out) *out=result;
    return result;
}
