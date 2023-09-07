#!/usr/bin/env python
# coding: utf-8

# In[11]:


tasks = []

def view_task():
    if not tasks:
        print("Your task list is empty.")
    else:
        for i, task in enumerate(tasks, 1):
            if task["completed"]:
                status = "(Completed)"
            else:
                status = "(Not completed)"
            print(f"{i}. {task['description']} {status}")
            
def add_task():
    task_description = input("Enter the task: ")
    tasks.append({"description": task_description, "completed": False})
    print("Task added successfully.")

def delete_task():
    if not tasks:
        print("Your task list is empty.")
    else:
        view_task()
        task_index = int(input("Enter the task number to delete: "))
        if 1 <= task_index <= len(tasks):
            del tasks[task_index - 1]
            print("Task deleted successfully.")
        else:
            print("Invalid task index.")

def mark_completed():
    if not tasks:
        print("Your task list is empty.")
    else:
        view_task()
        task_index = int(input("Enter the task number to mark as completed: "))
        if 1 <= task_index <= len(tasks):
            tasks[task_index - 1]["completed"] = True
            print("Task marked as completed.")
        else:
            print("Invalid task index.")

def save_to_file():
    filename = input("Enter the filename to save the list: ")
    with open(filename, "w") as file:
        for task in tasks:
            file.write(f"{task['description']},{task['completed']}\n")




# In[12]:


view_task()


# In[13]:


add_task()


# In[14]:


add_task()


# In[15]:


view_task()


# In[16]:


mark_completed()


# In[17]:


view_task()


# In[18]:


delete_task()


# In[19]:


view_task()


# In[20]:


save_to_file()


# In[21]:


add_task()


# In[23]:


mark_completed()


# In[24]:


save_to_file()


# In[ ]:




