#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Get the input fronmthe user if his/her age above 20 she is eligible for voting and you want to put certain contrain.
age=int(input("Enter the age:"))
if age >20 and age<=120:
    print("He/She is eligible for the voting")
else:
    print("He/She is not eligible for voting")


# In[1]:


'''write a program to print the below number sequence 
55555
4444
333
22
1'''
num_rows = 5
start_value = 5
current_num = 5

for i in range(num_rows):
    for j in range(start_value):
        print(current_num, end=' ')
    print()
    start_value -= 1
    current_num -= 1


# In[2]:


'''write a program to print a below sequence
54321
5432
543
54
5'''
num_rows = 5
start_value = 5
current_num = 5

for i in range(num_rows):
    for j in range(start_value):
        print(current_num, end=' ')
        current_num -= 1
    print()
    current_num = 5
    start_value -= 1


# In[3]:


'''write a program to print sequence given below
*****
****
***
**
*'''
num_rows = 5

for i in range(num_rows, 0, -1):
    for j in range(1, i + 1):
        print('*', end=' ')
    print()


# In[5]:


'''write a program to print sequence given below
*
**
***
****
*****'''
num_rows = 5

for row in range(1, num_rows + 1):
    for col in range(1, row + 1):
        print('*', end=' ')
    print()


# In[6]:


num_rows = 5
stars = 1
spaces = num_rows - 1

for i in range(num_rows):
    for j in range(spaces):
        print(' ', end=' ')
    for k in range(stars):
        if k % 2 == 0:
            print('*', end=' ')
        else:
            print(' ', end=' ')
    print()
    stars += 2
    spaces -= 1


# In[7]:


num_rows = 5
spaces = num_rows // 2
stars = 1

for i in range(num_rows):
    for j in range(spaces):
        print(' ', end=' ')
    for k in range(stars):
        print('*', end=' ')
    print()
    if i < num_rows // 2:
        stars += 2
        spaces -= 1
    else:
        stars -= 2
        spaces += 1


# In[18]:


s1=[1,2,3]
s2=[1,2,3]
print(s1 is s2)


# In[19]:


s1=[1,2,3]
s2=[1,2,3]
s1=s2
print(s1 is s2)


# In[ ]:




