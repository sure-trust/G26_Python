#!/usr/bin/env python
# coding: utf-8

# # PERSONAL FINANCE MANAGER
# 

# In[18]:


import sys
class Finance_Manager():
    def __init__(self):
        self.income=0
        self.incomesource=[]
        self.incomelist=[]
        self.expenses=0
        self.expensetype=[]
        self.expenseslist=[]
        self.promptincome()
    def income_info(self):
        ask_income=input('Add income? [yes/no]:')
        return ask_income
    def salary(self):    
        self.income=sum(self.incomelist)
    def expense_info(self):
        ask_expense=input('Add expense? [yes/no]:')
        return ask_expense
    def expenditure(self):
        self.expenses=sum(self.expenseslist)
    def income_check(self):
        if not self.incomelist:
            print('Please enter atleast one source of income:')
            self.promptincome()
        else:
            return
    def expense_check(self):
        if not self.expenseslist:
            print('please enter atleast one expense:')
            self.promptexpenses()
        else:
            return
    def promptincome(self):
        x=False
        while not x:
            result= self.income_info()
            if result=='yes':
                income_input=int(input('Enter amount of income:'))
                self.incomelist.append(income_input)
                incomesource=input('Enter income source:')
                self.incomesource.append(incomesource)
            else:
                self.income_check()
                x=True
        self.salary()
        name=[name for name in self.incomesource]
        income=[income for income in self.incomelist]
        incomedict=dict(zip(name,income))
        for i in incomedict:
            print(i + ':','RS.'+str(incomedict[i]))
        print('Total income of a employee is:','RS.'+str(self.income))
        self.promptexpenses()
    def promptexpenses(self):
        x=False
        while not x:
            result=self.expense_info()
            if result=='yes':
                expense_input=int(input('Enter expenses amount:'))
                self.expenseslist.append(expense_input)
                expensetype=input('Enter expense types:')
                self.expensetype.append(expensetype)
            else:
                self.expense_check()
                x=True
        self.expenditure()
        name=[name for name in self.expensetype]
        expense=[expense for expense in self.expenseslist]
        expensedict=dict(zip(name,expense))
        for i in expensedict:
            print(i+ ':','Rs.'+str(expensedict[i]))
        print('Total expenses of the person is:','RS.'+str(self.expenses))
        self.uservalue()
    def uservalue(self):
        valout=self.income-self.expenses
        if valout<0:
            print('You are in negative,you have a deficit of '+'RS.'+str(valout))
        if valout==0:
            print('you have broken,your expenses is as much as of your income')
        if valout>0:
            print('you are in the positive,you are having some savings')
        self.close_program()
    def close_program(self):
        print('Exiting the programme.')
        sys.exit()
f=Finance_Manager() 
f.income_info()
f.salary()
f.expense_info()
f.expenditure()
f.income_check()
f.expense_check()
f.promptincome()
f.promptexpenses()
f.uservalue()
f.close_program()

