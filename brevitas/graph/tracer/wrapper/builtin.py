class IntWrapper(int):
    def __new__(cls, value):
        x = int.__new__(cls, value)
        return x


class FloatWrapper(float):
    def __new__(cls, value):
        x = float.__new__(cls, value)
        return x


class StrWrapper(str):
    def __new__(cls, value):
        x = str.__new__(cls, value)
        return x

