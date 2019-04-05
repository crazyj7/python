import enum

class Fruit(enum.Enum):
	APPLE=0 
	BANANA=1 
	GRAPE=2

print(Fruit.APPLE)
print(Fruit.BANANA.value)
print(Fruit.GRAPE.value)

