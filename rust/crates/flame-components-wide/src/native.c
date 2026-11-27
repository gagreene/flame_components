#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

/* Rust supplies initialized, unaligned 16-byte buffers. Only copy value bytes
 * from C objects: x87 long double may have up to six indeterminate pad bytes. */
#define CAPACITY 16
#if CHAR_BIT != 8 || FLT_RADIX != 2
#error Unsupported long double representation
#elif LDBL_MANT_DIG == 53 && LDBL_MIN_EXP == -1021 && LDBL_MAX_EXP == 1024
#define VALUE_BYTES 8
#define NATIVE_MATH(name) name
#elif LDBL_MANT_DIG == 64 && LDBL_MIN_EXP == -16381 && LDBL_MAX_EXP == 16384 && (defined(__i386__) || defined(__x86_64__))
#define X87_EXTENDED 1
#define VALUE_BYTES 10
#define NATIVE_MATH(name) name##l
#elif LDBL_MANT_DIG == 113 && LDBL_MIN_EXP == -16381 && LDBL_MAX_EXP == 16384
#define VALUE_BYTES 16
#define NATIVE_MATH(name) name##l
#else
#error Unsupported long double representation
#endif

/* A typedef assertion works with older MSVC C language modes. */
#ifdef X87_EXTENDED
typedef char native_size_check[(sizeof(long double) >= VALUE_BYTES &&
                                sizeof(long double) <= CAPACITY) ? 1 : -1];
#else
typedef char native_size_check[sizeof(long double) == VALUE_BYTES ? 1 : -1];
#endif

static long double load_value(const unsigned char *input) {
    long double value = 0.0L;
    memcpy(&value, input, VALUE_BYTES);
    return value;
}

static void store_value(unsigned char *output, long double value) {
    memcpy(output, &value, VALUE_BYTES);
}

size_t fc_wide_format(int *precision, int *min_exp, int *max_exp) {
    *precision = LDBL_MANT_DIG;
    *min_exp = LDBL_MIN_EXP - 1;
    *max_exp = LDBL_MAX_EXP;
    return sizeof(long double);
}

void fc_wide_import(unsigned char *output, const unsigned char *input) {
    memcpy(output, input, VALUE_BYTES);
#ifdef X87_EXTENDED
    /* x87 stores its integer bit explicitly. Check the raw representation
     * before a C floating-point load can evaluate an unsupported encoding. */
    {
        unsigned int exponent = ((input[9] & 0x7f) << 8) | input[8];
        int integer_bit = input[7] & 0x80;
        if (exponent != 0 && !integer_bit) {
            store_value(output, (long double)NAN);
        } else if (exponent == 0 && integer_bit) {
            /* A pseudo-denormal has the same value as exponent one. */
            output[8] = 1;
        }
    }
#endif
}

void fc_wide_from_f64(unsigned char *output, double value) {
    store_value(output, (long double)value);
}

double fc_wide_to_f64(const unsigned char *input) {
    return (double)load_value(input);
}

float fc_wide_to_f32(const unsigned char *input) {
    return (float)load_value(input);
}

#define DEFINE_UNARY(name, expression)                                  \
    void fc_wide_##name(unsigned char *out, const unsigned char *in) {   \
        long double a = load_value(in);                                 \
        store_value(out, (expression));                                 \
    }

#define DEFINE_BINARY(name, expression)                                 \
    void fc_wide_##name(unsigned char *out, const unsigned char *left,   \
                        const unsigned char *right) {                   \
        long double a = load_value(left);                               \
        long double b = load_value(right);                              \
        store_value(out, (expression));                                 \
    }

DEFINE_BINARY(add, a + b)
DEFINE_BINARY(sub, a - b)
DEFINE_BINARY(mul, a * b)
DEFINE_BINARY(div, a / b)
DEFINE_BINARY(pow, NATIVE_MATH(pow)(a, b))
DEFINE_UNARY(sin, NATIVE_MATH(sin)(a))
DEFINE_UNARY(sqrt, NATIVE_MATH(sqrt)(a))
DEFINE_UNARY(log, NATIVE_MATH(log)(a))
DEFINE_UNARY(atan, NATIVE_MATH(atan)(a))
DEFINE_UNARY(asin, NATIVE_MATH(asin)(a))
DEFINE_UNARY(acos, NATIVE_MATH(acos)(a))
DEFINE_UNARY(abs, NATIVE_MATH(fabs)(a))

/* NumPy rounds the conversion factor in the input dtype before multiplying. */
#define PI 3.141592653589793238462643383279502884L
DEFINE_UNARY(radians, a * (PI / 180.0L))
DEFINE_UNARY(degrees, a * (180.0L / PI))

int fc_wide_is_nan(const unsigned char *input) {
    return isnan(load_value(input));
}

int fc_wide_is_finite(const unsigned char *input) {
    return isfinite(load_value(input));
}

int fc_wide_compare(const unsigned char *left, const unsigned char *right) {
    long double a = load_value(left);
    long double b = load_value(right);
    if (isnan(a) || isnan(b)) {
        return 2;
    }
    return (a > b) - (a < b);
}
