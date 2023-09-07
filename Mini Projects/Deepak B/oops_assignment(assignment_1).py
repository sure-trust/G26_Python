#!/usr/bin/env python
# coding: utf-8

# # Problem1

# In[2]:


class Document:
    Title="One_Piece"
    auther="Eiichiro_Oda"
class Book(Document):
    year=1997
class Article(Document):
    pages=562
    
    

        


# In[3]:


book1=Book()


# In[4]:


book1.auther


# In[5]:


book1.pages


# In[6]:


book2=Article()


# In[7]:


book2.pages


# In[8]:


book2.Title


# # Problem2

# In[30]:


class Person:
    Name ="Deepak.B"
    age =21

class Student:
    student_id ="1DB20EC024"

    def display_info(self):
        print(f"Name:{Person.Name}")
        print(f"Age:{Person.age}")
        print(f"Student ID:{self.student_id}")


# In[31]:


student1=Student()


# In[32]:


student1.display_info()


# # Problem 3

# In[38]:


class Bank_account:
    def deposit(self):
        pass
    def withdraw(self):
        pass
class Savings_account(Bank_account):
    def __init__(self,in_amount,fi_amount):
        self.in_amount=in_amount
        self.fi_amount=fi_amount
    def deposit(self):
        return self.in_amount+self.fi_amount
    def withdraw(self):
        return self.in_amount-self.fi_amount
class Checking_account(Bank_account):
    def __init__(self,o_amount, p_amount):
        self.o_amount = o_amount
        self.p_amount = p_amount
    def deposit(self):
        return self.o_amount*self.p_amount
    def withdraw(self):
        return self.o_amount/ self.p_amount   


# In[42]:


savings = Savings_account(1000,500)
checking = Checking_account(2000,100)


# In[43]:


print("Savings Account-Deposit:", savings.deposit())
print("Savings Account-Withdraw:", savings.withdraw())
print("Checking Account-Deposit:", checking.deposit())
print("Checking Account-Withdraw:", checking.withdraw())


# In[1]:


#Inheritance is a fundamental concept in object-oriented programming and is used extensively in the real world. 
#There are several types of inheritance, including single inheritance, multiple inheritance, multilevel inheritance, 
#hierarchical inheritance, and hybrid inheritance1. The complexity of these types of inheritance varies depending on the 
#specific use case and the relationships between the classes involved. For example, single inheritance, 
#where a subclass inherits from one superclass, is relatively simple. On the other hand, multiple inheritance, 
#where a subclass inherits from two or more superclasses, can be more complex due to potential conflicts between 
#inherited methods and attributes2. In general, it’s important to carefully design class hierarchies to ensure that 
#inheritance is used effectively and efficiently.

