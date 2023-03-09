import numpy as np

# Type enumeration
STRING = str
BOOL = bool
INT32 = np.int32
INT64 = np.int64
INT = int
UINT32 = np.uint32
UINT64 = np.uint64
UINT = UINT64
FLOAT32 = np.float32
FLOAT64 = np.float64
FLOAT = float

map_type = {STRING: STRING, BOOL: BOOL,
            INT32: INT32, INT64: INT64,
            INT: INT64,
            UINT32: UINT32, UINT64: UINT64,
            UINT: UINT64,
            FLOAT32: FLOAT32, FLOAT64: FLOAT64,
            FLOAT: FLOAT64,
            None: None}

__int_size = int(str(np.iinfo(int).dtype)[-2:])
if __int_size == 32:
    map_type[INT] = INT32
__uint_size = int(str(np.iinfo(int).dtype)[-2:])
if __uint_size == 32:
    _map_type[UINT] = UINT32
__float_size = int(str(np.finfo(float).dtype)[-2:])
if __float_size == 32:
    map_type[FLOAT] = FLOAT32
assert __int_size in [32, 64]
assert __float_size in [32, 64]
__io_float_map = {64: "double", 32: "float"}
__io_int_map = {64: "int64_t", 32: "int32_t"}
__io_uint_map = {64: "uint64_t", 32: "uint32_t"}

# Types allowed for configuring the library
ALLOWED_INT_TYPES = {INT32: "i32", INT64: "i64"}  # , int: "i%d" % __int_size}
ALLOWED_UINT_TYPES = {UINT32: "u32", UINT64: "u64"} #, uint: "i%d" % __int_size}
ALLOWED_FLOAT_TYPES = {FLOAT32: "f32", FLOAT64: "f64"}  # , float: "f%d" % __float_size}

# Types allowed to be pushed/fetched
ALLOWED_IO_TYPES = {FLOAT32: "float", FLOAT64: "double",
                    INT32: "int32_t", INT64: "int64_t",
                    UINT32: "uint32_t", UINT64: "uint64_t",
                    FLOAT: __io_float_map[__float_size],
                    UINT: __io_uint_map[__int_size],
                    INT: __io_int_map[__int_size],
                    str: "string"}


def get_int_type_str(typein):
    if typein in ALLOWED_INT_TYPES.keys():
        return ALLOWED_INT_TYPES[typein]
    else:
        raise Exception("Integer type '{}' not supported. Supported types : [int, np.int32, np.int64]".format(typein))

def get_uint_type_str(typein):
    if typein in ALLOWED_UINT_TYPES.keys():
        return ALLOWED_UINT_TYPES[typein]
    else:
        pass

def get_float_type_str(typein):
    if typein in ALLOWED_FLOAT_TYPES.keys():
        return ALLOWED_FLOAT_TYPES[typein]
    else:
        raise Exception("Float type '{}' not supported. "
                        "Supported types : [float, np.float32, np.float64]".format(typein))

def get_io_type_str(typein):
    if typein in ALLOWED_IO_TYPES.keys():
        return ALLOWED_IO_TYPES[typein]
    else:
        raise Exception("Float type '{}' not supported. "
                        "Supported types : [float, np.float32, np.float64]".format(typein))


def safe_cast(value_type, value):
    if not isinstance(value, str):  # Only for numerics types
        if not np.can_cast(value, value_type):
            raise Exception("Value '{}' cannot be safely casted to type '{}'.".format(value, value_type.__name__))
    return value_type(value)
