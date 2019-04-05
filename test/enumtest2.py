
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


Fruit = enum('APPLE','BANANA','GRAPE')

print(Fruit.APPLE)
print(Fruit.BANANA)
print(Fruit.GRAPE)

