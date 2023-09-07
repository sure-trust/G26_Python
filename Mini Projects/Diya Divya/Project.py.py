#!/usr/bin/env python
# coding: utf-8

# ### Develop a program that helps roommates or friends split expenses and keep track os shared bills. Calculate each persons share and generate a report

# In[1]:


class splitexpenses:
    def __init__(self):
        self.users = {}
        self.expenses = {}
    def add_user(self, user_name):
        if user_name not in self.users:
            self.users[user_name] = 0
            self.expenses[user_name] = []
    def add_expense(self, user_name, expense_amount):
        if user_name in self.users:
            self.expenses[user_name].append(expense_amount)
        else:
            print(f"User '{user_name}' does not exist. Please add the user first.")

    def calculate_shares(self):
        total_expense = sum(sum(user_expenses) for user_expenses in self.expenses.values())
        num_users = len(self.users)
        share_amount = total_expense / num_users

        for user_name in self.users:
            user_total_expense = sum(self.expenses[user_name])
            self.users[user_name] = share_amount - user_total_expense

    def generate_report(self):
        print("\nExpense Report:")
        for user_name, share in (self.users).items():
            print(f"{user_name}: {share:.2f}")

    def run(self):
        while True:
            choice = input("Enter your choice (1-5): ")

            if choice == "1":
                user_name = input("Enter user name: ")
                self.add_user(user_name)
            elif choice == "2":
                user_name = input("Enter user name: ")
                expense_amount = float(input("Enter expense amount: "))
                self.add_expense(user_name, expense_amount)
            elif choice == "3":
                self.calculate_shares()
                print("Shares calculated.")
            elif choice == "4":
                self.generate_report()
            elif choice == "5":
                print("Exiting.")
                break
            else:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    splitter = splitexpenses()
    splitter.run()


# In[ ]:




