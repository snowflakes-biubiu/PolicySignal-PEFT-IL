import sys

print("Python output encoding:", sys.stdout.encoding)
print("Chinese test: 类别统计、正确率、错误转移")
print([hex(ord(c)) for c in "类别统计"])