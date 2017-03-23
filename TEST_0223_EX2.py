
# by convention, we give classes PascalCase names
class Set:

 def __init__(self, values=None):
    self.dict = {}
    if values is not None:
         for value in values:
             self.add(value)

 def __repr__(self):
        """this is the string representation of a Set object
        if you type it at the Python prompt or pass it to str()"""
        return "Set: " + str(self.dict.keys())

 # we'll represent membership by being a key in self.dict with value True
 def add(self, value):
     self.dict[value] = True

 # value is in the Set if it's a key in the dictionary
 def contains(self, value):
     return value in self.dict

 def remove(self, value):
     del self.dict[value]

s = Set([1,2,3])
s.add(4)
print s.contains(4) # True
s.remove(3)
print s.contains(3) # False
print("Str(Set):"+str(s))


class Set1(Set):
     def __repr__(self, values=None):
        return "My Data in Set: "+str(self.dict.keys())
s2=Set1([1,2,3,4])

print("Str(Set1):" + str(s2))