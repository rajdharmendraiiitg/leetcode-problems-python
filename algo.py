""" def carPooling( trips,capacity):
        lst = []
        for n, start, end in trips:
            lst.append((start, 
                        n))
            lst.append((end, -n))
        lst.sort()
        print(lst)
        pas = 0
        for loc in lst:
            pas += loc[1]
            if pas > capacity:
                print(pas)
                return False
        return True
print(carPooling([[2,1,5],[3,3,7]],4)) """

# A Sample class with init method
a = 9
class Person:
    
	# init method or constructor
	def __init__(self, name):
		self.name = name

	# Sample Method
	def say_hi(self):
		print('Hello, my name is', self.name,a)

# Creating different objects
p1 = Person('nikhil')
p2 = Person('Abhinav')
p3 = Person('Anshul')

p1.say_hi()
p2.say_hi()
p3.say_hi()