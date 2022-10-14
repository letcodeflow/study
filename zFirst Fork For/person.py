class Person:
    def __init__(self,name,age,add):
        self.name = name
        self.age = age
        self.add = add

    def greeting(self):
        print('hi {0}'.format(self.name))